#!/usr/bin/env python3

if __name__ == '__main__':

    import os
    import shutil
    import argparse
    
    parser = argparse.ArgumentParser('Merge the Split Data into Uni-class and Multi-class Subsets \
                                     for the Subsampling Training Strategy', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to output directory.')
    args = parser.parse_args()
  
    label_dir = os.path.join(args.out_dir, "class_labels")
    mask_dir = os.path.join(args.out_dir, "masks")

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    uni_image_dir = os.path.join(args.out_dir, "images", "uni_class")
    multi_image_dir = os.path.join(args.out_dir, "images", "multi_class")

    os.makedirs(uni_image_dir, exist_ok=True)
    os.makedirs(multi_image_dir, exist_ok=True)

    if os.path.exists(args.data_dir) and os.path.isdir(args.data_dir):
        
        for dir in os.listdir(args.data_dir):

            in_path = os.path.join(args.data_dir, dir, 'class_labels')
            for file_name in os.listdir(in_path):
                shutil.copy(os.path.join(in_path, file_name), label_dir)
            
            in_path = os.path.join(args.data_dir, dir, 'masks')
            for file_name in os.listdir(in_path):
                shutil.copy(os.path.join(in_path, file_name), mask_dir)
            
            in_path = os.path.join(args.data_dir, dir, 'images')
            for file_name in os.listdir(in_path):
                shutil.copy(os.path.join(in_path, file_name), 
                    uni_image_dir if dir=='1' else multi_image_dir)
        
    else:
        raise OSError('Not a valid input directory!')


# ============ Output Directory Structure ============
# .
# ├── images
# │   ├── uni_class
# │   │   ├── file_name_1.tiff
# │   │   ├── file_name_2.tiff
# │   │    ⋮
# │   │   └── file_name_n.tiff
# │   └── multi_class
# │       ├── file_name_n+1.tiff
# │       ├── file_name_n+2.tiff
# │        ⋮
# │       └── file_name_N.tiff
# ├── class_labels
# │   ├── file_name_1.csv
# │   ├── file_name_2.csv
# │    ⋮
# │   └── file_name_N.csv
# └── masks
#     ├── file_name_1.tiff
#     ├── file_name_2.tiff
#      ⋮
#     └── file_name_N.tiff
#
# ====================================================
