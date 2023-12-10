# Visualize evolution of pseudomasks over different training epochs

import os, sys
import argparse
import csv, numpy
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == '__main__':

    def mask2rgb(idx_img):
        rgb_img = numpy.empty(idx_img.shape+(3,), dtype='uint8')
        for k,v in cmap.items():
            rgb_img[idx_img==k] = v
        return rgb_img
    
    parser = argparse.ArgumentParser('Pseudomask Evaluation', add_help=False)
    parser.add_argument('--pseudomask_dir', type=str, required=True,
                        help='Path to pseudomasks generated during training.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training dataset.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to save visualizations.')
    parser.add_argument('--cmap_path', type=str, required=True,
                        help='Path to color map file.')
    args = parser.parse_args()

    if args.cmap_path.endswith('.csv') and os.path.isfile(args.cmap_path):
        with open(args.cmap_path, newline='') as cmap_file:
            csv_reader = csv.reader(cmap_file, quoting=csv.QUOTE_NONNUMERIC)
            cmap = {row[0]: (row[1], row[2], row[3]) for row in csv_reader}
    else:
        sys.exit('Colormap Invalid!')

    os.makedirs(args.out_dir, exist_ok=True)

    pseudomask_dirs = [os.path.join(args.pseudomask_dir,subdir) 
        for subdir in os.listdir(args.pseudomask_dir) 
            if os.path.isdir(os.path.join(args.pseudomask_dir,subdir))]
    pseudomask_dirs = sorted(pseudomask_dirs, key = lambda dir: int(dir.split('epoch_')[1]))
    pseudomask_dirs.extend([args.pseudomask_dir, os.path.join(args.data_dir,'masks')])
    pseudomask_dirs.insert(0, os.path.join(args.data_dir,'images','multi_class'))

    num_dirs = len(pseudomask_dirs)
    image_files = [file for file in os.listdir(pseudomask_dirs[0]) if file.endswith('.tiff')]

    titles = {0: 'Input', num_dirs-1: 'GT', num_dirs-2: 'Final'}
    for file in image_files:
        plt.figure(figsize=(24, 8))
        
        for i, pseudomask_dir in enumerate(pseudomask_dirs):

            if 0 < i < num_dirs - 1:
                image_path = os.path.join(pseudomask_dir, file[:-4]+'png')
            else: 
                image_path = os.path.join(pseudomask_dir, file)

            image = numpy.array(Image.open(image_path))
            image, plt_cmap = (image, 'gray') if i==0 else (mask2rgb(image), 'viridis')
             
            plt.subplot(1, num_dirs, i+1)
            plt.imshow(image, aspect='equal', cmap=plt_cmap)
            plt.title(titles.get(i,pseudomask_dir.split(os.sep)[-1]))
            plt.axis('off')
            
        plt.savefig(os.path.join(args.out_dir,file), bbox_inches='tight', pad_inches=1)
        plt.close()
    
    print('fin.')
