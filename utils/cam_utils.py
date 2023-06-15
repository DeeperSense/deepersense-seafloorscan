import numpy
import torch

import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader as DL

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

from multiprocessing import Pool
from utils.logger import MetricLogger

from tqdm import tqdm


def update_pseudo_labels(model, img_size, batch_size, num_workers,
        cls_data_loader, seg_dataset, mode='train'):
    
    # item[img]: list of N tensors of shape (B,2,C,H,W) where N == #scales
    # item[label].shape == (B,K) and item[idx].shape == (B,1)

    def generate_and_store(cls_data_loader, seg_dataset):
        with Pool(processes=num_workers) as pool:
            for item in tqdm(cls_data_loader):
                idx = item["idx"]
                img_i = [img_ii.cuda(non_blocking=True) for img_ii in item["img"]]
                res = generate_pseudo_labels(model, img_i, item["label"], img_size)
                results = [(idx[i].cpu().item(), res[0][i], res[1][i]) for i in range(batch_size)]
                if len(results) > 0:
                    pool.map(seg_dataset.update_cam, results)
    
    model.eval()
    with torch.set_grad_enabled(False):
        if mode == 'eval':
            print("\nCreating Pseudo Labels for evaluation...")
            generate_and_store(cls_data_loader, seg_dataset)
        else:  
            print("\nCreating Pseudo Labels for training...")
            generate_and_store(cls_data_loader['train'], seg_dataset['train'])
            print("\nCreating Pseudo Labels for validation...")
            generate_and_store(cls_data_loader['val'], seg_dataset['val'])


def generate_pseudo_labels(model, imgs, cls_label_true, original_image_size):
    
    # reshape from (B,2,C,H,W) to ((B,C,H,W),(B,C,H,W))
    def reshape(img):
        img1, img2 = torch.chunk(img, 2, dim=1)
        img1 = torch.squeeze(img1, dim=1)
        img2 = torch.squeeze(img2, dim=1)
        return (img1,img2)
    
    with torch.set_grad_enabled(False):
        cams = [model.forward_cam(reshape(img)) for img in imgs]

        strided_cam = [
            F.interpolate(
                cam,                    # (B,K,h,w)
                original_image_size,    # (H,W)
                mode="bilinear",
                align_corners=False,
            )
            for cam in cams
        ] # list of N tensors of shape (B,K,H,W) where N == #scales

        strided_cam = torch.sum(torch.stack(strided_cam, 0), 0) # (B,K,H,W)

        # list of B tuples of varying length | length = count(K==1)
        valid_cat = [tuple(idx[0] for idx in row.nonzero().tolist())
                        for row in cls_label_true]  # cls_label_true.shape == (B,K)

        cams_list = []
        keys_list = []

        for b in range(strided_cam.shape[0]):
            valid_cams = strided_cam[b, valid_cat[b]]   # (V,H,W) where V = count(K==1)

            valid_keys = valid_cat[b]                   # tuple of length V

            valid_cams /= (F.adaptive_max_pool2d(valid_cams, (1, 1)) + 1e-5)    # normalize

            cams_list.append(valid_cams)
            keys_list.append(valid_keys)

        return cams_list, keys_list         # lengths == B


def crf_inference_label(img, labels, t=10, n_labels=4, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_labels(
        labels, n_labels, gt_prob=gt_prob, zero_unsure=False
    )

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    
    pairwise_energy = create_pairwise_bilateral(sdims=(20, 20), schan=5, chdim=2,
                            img=numpy.ascontiguousarray(numpy.copy(img)))
    d.addPairwiseEnergy(pairwise_energy, compat=10)

    q = d.inference(t)

    return numpy.argmax(numpy.array(q).reshape((n_labels, h, w)), axis=0)


def calculate_pseudo_miou(dataset, num_classes, batch_size, num_workers):
    metric_logger = MetricLogger(delimiter="  ")

    dataloader = DL(dataset, batch_size=batch_size, num_workers=num_workers, 
            persistent_workers=True, drop_last=True, pin_memory=True)

    for item in dataloader:
        pseudo_mask, gt_mask = (item["pseudo_mask"], item["gt_mask"])
    
        pseudo_mask = F.one_hot(pseudo_mask.long(), num_classes).permute(0,3,1,2)
        gt_mask = F.one_hot(gt_mask.long(), num_classes).permute(0,3,1,2)
   
        inter = torch.sum(torch.logical_and(pseudo_mask, gt_mask), dim=(2,3))   # B x K
        union = torch.sum(torch.logical_or(pseudo_mask, gt_mask), dim=(2,3))    # B x K
        miou = torch.mean((inter+1e-6)/(union+1e-6), dim=(0,1))

        metric_logger.update(mask_miou=miou)
    
    return metric_logger.meters['mask_miou'].global_avg
