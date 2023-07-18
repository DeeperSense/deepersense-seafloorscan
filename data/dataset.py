import os
import csv
import glob
import random

from PIL import Image
from imageio import imwrite

import numpy
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import data.augmentations as aug

from utils.cam_utils import crf_inference_label


class ClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=ToTensor()):
        self.transform = transform

        self.img_files = glob.glob(os.path.join(data_dir,'images','*.tiff'))
        self.file_names = [os.path.splitext(os.path.basename(img_path))[0]
                                for img_path in self.img_files]
        
        self.class_label_dir = os.path.join(data_dir,'class_labels')
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert('L')
        img, transform_params = self._apply_transforms(img)
        
        file_name = self.file_names[idx]
        label = self._read_csv(file_name)

        return {"name": file_name, "idx": idx, "img": img, "label":label,
                    "transform_params": transform_params}
    
    def _read_csv(self, file_name):
        with open(os.path.join(self.class_label_dir,file_name+'.csv'), "r") as csv_file:
            reader = csv.reader(csv_file)
            next(reader) # skip header
            data = torch.tensor([int(row[1]) for row in reader])
        return data
    
    def _apply_transforms(self, image):
        transform_params = []
        transformed_image = image
        for transform in self.transform.transforms:
            transformed_image, transform_param = transform(transformed_image, getParam=True)
            if transform.TYPE == aug.TransformType.GEOMETRIC:
                transform_params.append(transform_param)
        return transformed_image, transform_params


class ClassificationDatasetMSF(ClassificationDataset):
    def __init__(self, data_dir, scales=(1.0,), transform=ToTensor()):
        super().__init__(data_dir, transform=transform)
        self.scales = scales
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert('L')

        file_name = self.file_names[idx]
        label = self._read_csv(file_name)

        ms_img_list = []
        random_seed = random.randint(1, 65536)  # random seed for each call

        for scale in self.scales:
            s_img = F.resize(img, size=(int(img.size[0]*scale), int(img.size[1]*scale))) \
                        if scale!=1 else img
            
            numpy.random.seed(random_seed)
            random.seed(random_seed)

            s_img, transform_params = self._apply_transforms(s_img)
            ms_img_list.append(torch.stack([s_img, torch.flip(s_img, dims=(-1,))], dim=0))

        return {"name": file_name, "idx": idx, "img": ms_img_list, "label":label, 
                    "transform_params": transform_params}


class PseudoSegmentationDataset(ClassificationDataset):
    def __init__(self, data_dir, pseudo_mask_dir=None, num_classes=1, transform=None):
        super().__init__(data_dir, transform)
        self.num_classes = num_classes
        self.pseudo_mask_dir = pseudo_mask_dir
        self.gt_mask_dir = os.path.join(data_dir, 'masks')
        if not os.path.exists(self.pseudo_mask_dir):
            os.makedirs(self.pseudo_mask_dir)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert('L')

        file_name = self.file_names[idx]
        label = self._read_csv(file_name)
        pseudo_mask = self._get_pseudo_mask(file_name, img.size)
        
        try:
            gt_mask = self._get_gt_mask(file_name)
        except:
            gt_mask = pseudo_mask

        img, pseudo_mask, gt_mask = self.transform((img, pseudo_mask, gt_mask))

        return {"name": file_name, "idx": idx, "img": img, "label": label,
                    "pseudo_mask": pseudo_mask.copy(), "gt_mask": gt_mask.copy()}
    
    def _get_pseudo_mask(self, file_name, im_size):
        path = os.path.join(self.pseudo_mask_dir, file_name+'.png')
        return numpy.array(Image.open(path)) if os.path.exists(path) else \
                numpy.zeros((im_size[1],im_size[0])).astype(numpy.uint8)
    
    def _get_gt_mask(self, file_name):
        path = os.path.join(self.gt_mask_dir, file_name+'.tiff')
        label = numpy.array(Image.open(path))
        return label

    def _revert_transformations(self, image, transform_params):
        transformed_image = image
        for transform in self.transform.transforms[::-1]:
            if transform.TYPE == aug.TransformType.GEOMETRIC:
                transformed_image = transform(transformed_image, transform_params.pop())
        return transformed_image

    def update_cam(self, cam_info):
        idx, transform_params, cams, keys = cam_info
        
        cams = cams.cpu().numpy()                                                   # (V,H,W)
        cams = numpy.argmax(cams, axis=0)                                           # (H,W)

        img = numpy.asarray(Image.open(self.img_files[idx]).convert('L'))
        img = numpy.expand_dims(img, axis=-1)                                       # (H,W,C)

        pred = crf_inference_label(img, cams, n_labels=self.num_classes)            # (H,W)
        conf = numpy.zeros_like(pred)
        for i in range(len(keys)):
            conf[pred==i] = keys[i]

        path = os.path.join(self.pseudo_mask_dir, self.file_names[idx]+'.png')
        conf = self._revert_transformations(conf, transform_params)
        imwrite(path, conf.astype(numpy.uint8))
