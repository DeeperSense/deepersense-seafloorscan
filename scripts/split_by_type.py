#!/usr/bin/env python3

if __name__ == '__main__':

    import os
    import shutil
    import argparse
    import csv
    import numpy
    import matplotlib.pyplot as plt
    from collections import defaultdict

    class RatioCheck(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not 0 < values < 1:
                raise argparse.ArgumentError(self, "ratio must be between 0 and 1")
            setattr(namespace, self.dest, values)
    
    parser = argparse.ArgumentParser('Split Data by Number of Classes per Image', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to output directory.')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        metavar='{0..1}', action=RatioCheck,
                        help='Train-Val split ratio.')
    parser.add_argument('--plot_stats', action='store_true',
                        help='Plot stats of input data.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    mask_dir = os.path.join(args.data_dir,"masks")
    image_dir = os.path.join(args.data_dir,"images")
    label_dir = os.path.join(args.data_dir,"class_labels")

    cat_freq = defaultdict(int)
    cat_imgs = defaultdict(list)

    for label_file in [file for file in os.listdir(label_dir) if file.endswith('.csv')]:
        with open(os.path.join(label_dir, label_file), mode='r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)                            # skip header
            cat = [int(row[1]) for row in reader]   # one-hot
            cat.insert(0,numpy.sum(cat))            # count
            cat = tuple(cat)
            cat_freq[cat] += 1
            cat_imgs[cat].append(label_file[:-4])

    if args.plot_stats:
        plt.bar(*zip(*[(str(k),v) for k,v in sorted(
            cat_freq.items(), reverse=True, key=lambda item:item[1])]))
        plt.xticks(rotation=90)
        plt.xlabel('Categories')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(args.out_dir,'data_stats.png'),
            bbox_inches='tight', pad_inches=0.25)
        plt.close()

    for cat in cat_freq.keys():
        split_size = int(cat_freq[cat]*args.split_ratio)
        
        for i in range(split_size):
            file_name = cat_imgs[cat][i]
        
            out_path = os.path.join(args.out_dir, 'train', str(cat[0]), 'class_labels')
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(label_dir, file_name+'.csv'),
                            os.path.join(out_path, file_name+'.csv'))
            
            out_path = os.path.join(args.out_dir, 'train', str(cat[0]), 'images')
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(image_dir, file_name+'.tiff'),
                            os.path.join(out_path, file_name+'.tiff'))
            
            out_path = os.path.join(args.out_dir, 'train', str(cat[0]), 'masks')
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(mask_dir, file_name+'.tiff'),
                            os.path.join(out_path, file_name+'.tiff'))
        
        for i in range(split_size, cat_freq[cat]):
            file_name = cat_imgs[cat][i]
        
            out_path = os.path.join(args.out_dir, 'val', str(cat[0]), 'class_labels')
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(label_dir, file_name+'.csv'),
                            os.path.join(out_path, file_name+'.csv'))
            
            out_path = os.path.join(args.out_dir, 'val', str(cat[0]), 'images')
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(image_dir, file_name+'.tiff'),
                            os.path.join(out_path, file_name+'.tiff'))
            
            out_path = os.path.join(args.out_dir, 'val', str(cat[0]), 'masks')
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(mask_dir, file_name+'.tiff'),
                            os.path.join(out_path, file_name+'.tiff'))
