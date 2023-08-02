import os
import sys
import csv
import time

import argparse
import configs.models
import configs.base

import numpy
import torch

from models.build import build_model
from torch.nn.functional import one_hot

import data.augmentations as aug
from torchvision.transforms import Compose

from data.dataset import PseudoSegmentationDataset, ClassificationDatasetMSF
from torch.utils.data.dataloader import DataLoader as DL

from sklearn.metrics import average_precision_score
from utils.cam_utils import update_pseudo_labels
from torchinfo import summary

import matplotlib.pyplot as plt
import torch.multiprocessing as M

from utils import utils


def qualitative_eval(model, cls_data_loader, seg_data_loader, out_dir, device, cmap_path, num_images=4):
    """Qualitative comparison between expected and predicted class labels
    presented as a grid of `num_images` samples.
    """
    
    if cmap_path.endswith('.csv') and os.path.isfile(cmap_path):
        with open(cmap_path, newline='') as cmap_file:
            csv_reader = csv.reader(cmap_file, quoting=csv.QUOTE_NONNUMERIC)
            cmap = {row[0]: (row[1], row[2], row[3]) for row in csv_reader}
    else:
        sys.exit('Colormap Invalid!')
    
    def mask2rgb(idx_img):
        nonlocal cmap
        rgb_img = numpy.empty(idx_img.shape+(3,), dtype='uint8')
        for k,v in cmap.items():
            rgb_img[idx_img==k] = v
        return rgb_img

    def onehot2rgb(hot_img):
        idx_img = numpy.argmax(hot_img, axis=0)
        rgb_img = mask2rgb(idx_img)
        return rgb_img

    def imshow(img, gt_mask, pseudo_mask, pred, ax, n):
        img = img.squeeze(0).numpy()
        ax[n,0].imshow(img,'gray')
        ax[n,0].visible = False
        ax[n,0].axis('off')
        ax[n,1].imshow(mask2rgb(gt_mask))
        ax[n,1].visible = False
        ax[n,1].axis('off')
        ax[n,2].imshow(mask2rgb(pseudo_mask))
        ax[n,2].visible = False
        ax[n,2].axis('off')
        ax[n,3].imshow(onehot2rgb(pred))
        ax[n,3].visible = False
        ax[n,3].axis('off')
        if n==0:
            ax[n,0].set_title('Input', pad=24, size=24, weight='bold')
            ax[n,1].set_title('Noisy GT', pad=24, size=24, weight='bold')
            ax[n,2].set_title('Pseudo Mask', pad=24, size=24, weight='bold')
            ax[n,3].set_title('Prediction', pad=24, size=24, weight='bold')

    model.to(device)
    with torch.no_grad():

        update_pseudo_labels(model, config.DATA.IMAGE_SIZE, args.batch_size, args.num_workers*2,
            cls_data_loader, seg_data_loader.dataset, mode='eval')
        
        for i, batch in enumerate(seg_data_loader):

            inputs = batch["img"].to(device)
            outputs = model(inputs, mode='eval', isClassificationOnly=False)["seg"]

            gt_mask = batch["gt_mask"]
            pseudo_mask = batch["pseudo_mask"]

            if device == 'cuda':
                inputs = inputs.cpu()
                outputs = outputs.cpu()
                gt_mask = gt_mask.cpu()
                pseudo_mask = pseudo_mask.cpu()
            
            fig, ax = plt.subplots(num_images, 4, figsize=(20, 5*num_images))
            for j in range(inputs.size()[0]):
                imshow(inputs[j], gt_mask[j], pseudo_mask[j], outputs[j], ax, j%num_images)
                if j%num_images == num_images-1:
                    fig.savefig(os.path.join(out_dir,f'batch_{i}_report_{j//num_images}.png'), 
                                bbox_inches='tight', pad_inches=1)
                    plt.close(fig)


def quantitative_eval(model, data_loader, out_dir, device, num_classes):
    """A set of class-wise and average performance metrics saved in a text file.
    """
    
    eps = 1e-6
    ap_score = 0
    num_images = num_iter = 0
    num_px = torch.zeros(num_classes).to(device)
    acc = torch.zeros(num_classes).to(device)
    pre = torch.zeros(num_classes).to(device)
    rec = torch.zeros(num_classes).to(device)
    iou = torch.zeros(num_classes).to(device)

    iou_per_category = numpy.zeros(num_classes)
    num_images_per_category = numpy.zeros(num_classes)
    num_classes_per_category = {category: [0] * num_classes for category in range(1, num_classes + 1)}
    num_class_batch = None

    model.to(device)
    with torch.no_grad():
        for batch in data_loader:

            inputs = batch["img"].to(device)
            cls_label = batch["label"].to(device)
            seg_label = batch["gt_mask"].to(device).long()

            num_classes_batch =  torch.sum(cls_label, dim=1)
            num_images += inputs.size(0)
            num_iter += 1

            outputs = model(inputs, mode='eval', isClassificationOnly=False)
            seg_pred = torch.argmax(outputs["seg"], 1)
            cls_pred = torch.sigmoid(outputs["cls"]).cpu().numpy()

            seg_label = one_hot(seg_label, num_classes).permute(0,3,1,2)
            seg_pred = one_hot(seg_pred, num_classes).permute(0,3,1,2)

            inter = torch.sum(torch.logical_and(seg_pred, seg_label), dim=(2,3))
            union = torch.sum(torch.logical_or(seg_pred, seg_label), dim=(2,3))
            
            tp_fp = torch.sum(seg_pred, dim=(2,3))
            tp_fn = torch.sum(seg_label, dim=(2,3))

            batch_iou = (inter+eps)/(union+eps)
            for i in range(len(num_classes_batch)):
                num_images_per_category[num_classes_batch[i].item()-1] += 1 
                num_classes_per_category[num_classes_batch[i].item()] += \
                                                        numpy.array(cls_label[i].cpu().numpy())
                iou_per_category[num_classes_batch[i].item()-1] += torch.mean(batch_iou[i]).item()

            acc += torch.sum(inter, dim=0)
            pre += torch.sum((inter+eps)/(tp_fp+eps), dim=0)
            rec += torch.sum((inter+eps)/(tp_fn+eps), dim=0)
            iou += torch.sum((inter+eps)/(union+eps), dim=0)

            num_px += torch.sum(seg_label, dim=(0,2,3))

            cls_label = cls_label.detach().cpu().numpy()
            ap_score += average_precision_score(y_true=cls_label, y_score=cls_pred,
                                                    average='samples')
        
        acc /= num_px
        pre /= num_images
        rec /= num_images
        iou /= num_images
        dsc = (2*pre*rec)/(pre+rec)
        ap_score /= (num_iter+1)

        iou_per_category = numpy.array(iou_per_category)
        iou_per_category /= num_images_per_category
    
    with open(os.path.join(out_dir, 'eval_report.txt'), 'w') as out_file:
        print('Stats computed on %d test images\n' %num_images, file=out_file)

        print('Mean Accuracy of the network: %.4f%%' %(100*torch.mean(acc)), file=out_file)
        for i in range(num_classes):
            print('\tAccuracy of class %d : %.4f%%'
                  %(i, 100*acc[i]), file=out_file)
        print('\n', file=out_file)

        print('Mean Precision of the network: %.4f%%' %(100*torch.mean(pre)), file=out_file)
        for i in range(num_classes):
            print('\tPrecision of class %d : %.4f%%'
                  %(i, 100*pre[i]), file=out_file)
        print('\n', file=out_file)

        print('Mean Recall of the network: %.4f%%' %(100*torch.mean(rec)), file=out_file)
        for i in range(num_classes):
            print('\tRecall of class %d : %.4f%%'
                  %(i, 100*rec[i]), file=out_file)
        print('\n', file=out_file)

        print('Mean F1 Score of the network: %.4f%%' %(100*torch.mean(dsc)), file=out_file)
        for i in range(num_classes):
            print('\tF1 Score of class %d : %.4f%%'
                  %(i, 100*dsc[i]), file=out_file)
        print('\n', file=out_file)
        
        print('Mean IOU of the network: %.4f%%' %(100*torch.mean(iou)), file=out_file)
        for i in range(num_classes):
            print('\tIOU of class %d : %.4f%%'
                  %(i, 100*iou[i]), file=out_file)
        print('\n', file=out_file)

        print('AP Score of the network: %.4f%%' %ap_score, file=out_file)
        print('\n', file=out_file)

        print('Effect of number of classes on outputs', file=out_file)
        print('\n', file=out_file)

        print('Number of images per category', file=out_file)
        for i in range(len(num_images_per_category)):
            print('\t images with %d classes: %d'
                  %(i+1, num_images_per_category[i]), file=out_file)

        # save plot
        plt.bar([f"{i+1} Classes" for i in range(num_classes)], 
                 [num_images_per_category[i] for i in range(num_classes)])
        plt.ylabel('No. of Images')
        plt.title('Number of Images per Category')
        plt.savefig(os.path.join(out_dir, 'images_per_cat.png'))
        plt.close()
        
        print('\n', file=out_file)
        print('Mean IOU per category', file=out_file)
        for i in range(len(iou_per_category)-1):
            print('\t IOU with %d classes: %.4f%%'
                  %(i+1, 100*iou_per_category[i]), file=out_file)
            
        # save plot
        plt.bar([f"{i+1} Classes" for i in range(num_classes)], 
                [100*iou_per_category[i] for i in range(num_classes)])
        plt.ylabel('% MIOU')
        plt.title('MIOU per Category')
        plt.savefig(os.path.join(out_dir, 'miou_per_cat.png'))
        plt.close()


def runtime_eval(model, data_loader, out_dir, device, reps=25):
    """A set of metrics to gauge model efficiency in terms of throughput and
    parameter count. Saved in a text file.
    """
    assert device in ('cpu', 'cuda'), "Invalid device type!"
    
    model.to(device)
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["img"].to(device)
            batch_size = inputs.shape[0]

            # warmup
            for _ in range(10):
                _ = model(inputs, mode='infer', isClassificationOnly=False)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(reps):
                _ = model(inputs, mode='infer', isClassificationOnly=False)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            elapsed_time = (end-start)
            inference_speed = elapsed_time / (reps * batch_size)
            throughput = (reps * batch_size) / elapsed_time

            model_summary = summary(model, depth=1, input_data=inputs,
                col_names=("input_size","output_size","num_params","mult_adds"))
            break
    
    with open(os.path.join(out_dir, 'runtime_stats.txt'), 'w') as out_file:
        print('Stats averaged over %d reps' % reps, file=out_file)
        print(model_summary, file=out_file)
        print('Inference speed: %.4f s' % inference_speed, file=out_file)
        print('Throughput: %d images' % throughput, file=out_file)


if __name__ == '__main__':
    M.set_start_method('spawn')

    encoder_choices = ('sima_mini', 'sima_tiny')
    encoder_configs = {k: configs.models.__getattribute__(k) for k in encoder_choices}

    decoder_choices = ('symmetric', 'atrous')
    decoder_configs = {k: configs.models.__getattribute__(k) for k in decoder_choices}

    eval_choices = ('qualitative', 'quantitative', 'runtime')

    # ================ parsing arguments ================
    
    parser = argparse.ArgumentParser('S3Tseg Evaluation', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to evaluation data.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to configuration file.')
    parser.add_argument('--cmap_path', type=str, default=None,
                        help='Path to color map file.')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='Path to save logs.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to compute runtime statistics for (cpu | cuda).')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of distinct images per batch.')
    parser.add_argument('--encoder', type=str, default='sima_tiny', choices=encoder_choices,
                        help='Type of encoder architecture.')
    parser.add_argument('--decoder', type=str, default='atrous', choices=decoder_choices,
                        help='Type of decoder architecture.')
    parser.add_argument('--mode', type=str, default='visualize', choices=eval_choices,
                        help='Evaluation mode (qualitative | quantitative | runtime).')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ================ fetching model configuration ================

    config = configs.base._C.clone()                            # Base Configurations
    config.merge_from_list(encoder_configs[args.encoder]())     # Architecture defaults
    config.merge_from_list(decoder_configs[args.decoder]())
    if args.config_file:
        config.merge_from_file(args.config_file)                # User Customizations
    config.freeze()

    utils.fix_random_seeds(42)

    # ================ building model ================

    model = build_model(config)
    utils.load_pretrained_weights(model, args.model_path, 'eval')
    model.eval()

    print('model built')

    # ================ preparing data ================

    data_transforms = Compose([
            aug.ToTensor(),
            aug.Normalize(config.DATA.MEAN, config.DATA.STD)
        ])
    
    classification_dataset = ClassificationDatasetMSF(args.data_dir, config.TRAIN.SCALES, data_transforms)
    cls_data_loader = DL(classification_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=False, pin_memory=True, drop_last=True)
    
    segmentation_dataset = PseudoSegmentationDataset(args.data_dir, os.path.join(args.out_dir,'pseudo_masks'),
                                        config.DATA.NUM_CLASSES, data_transforms)
    seg_data_loader = DL(segmentation_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=False, pin_memory=True, drop_last=True)
    
    print('data ready')

    # ================ evaluation and reporting ================

    if args.mode == 'qualitative':
        qualitative_eval(model, cls_data_loader, seg_data_loader, args.out_dir, args.device, args.cmap_path)
    elif args.mode == 'quantitative':
        quantitative_eval(model, seg_data_loader, args.out_dir, args.device, config.DATA.NUM_CLASSES)
    elif args.mode == 'runtime':
        runtime_eval(model, seg_data_loader, args.out_dir, args.device)
    else:
        raise ValueError()
    
    print('reports saved')
