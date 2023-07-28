import os, sys
import argparse
import matplotlib.pyplot as plt
import csv, numpy, cv2
import matplotlib.pyplot as plt
from PIL import Image


def main(pseudomask_dir, data_dir, out_dir, cmap_path, mode):

    def get_numeric_part(directory):
        return int(directory.split("val_")[1])
    
    pseudomask_dir = f"{pseudomask_dir}/{mode}"
    gt_dir = f"{data_dir}/{mode}/masks"

    # get folders
    pseudomask_folders = [os.path.join(pseudomask_dir, folder) for folder in os.listdir(pseudomask_dir) if os.path.isdir(os.path.join(pseudomask_dir, folder))]

    # sort the folders
    pseudomask_folders = sorted(pseudomask_folders, key=get_numeric_part)

    # add validation and GT dir at the end
    pseudomask_folders.extend([pseudomask_dir, gt_dir])

    compare_pseudmasks(pseudomask_folders, out_dir, cmap_path)


def compare_pseudmasks(pseudomask_dirs, out_dir, cmap_path):

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

    folder_names = []

    # grab image files of the first directory
    image_files = [file for file in os.listdir(pseudomask_dirs[0]) if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))]
    
    for f in pseudomask_dirs:
        folder_names.append(f.split('/')[-1])

    num_folders = len(pseudomask_dirs)

    for file in image_files:
        plt.figure(figsize=(24, 8))
        for i in range(num_folders):

            # modify filename for gt extension (tiff)
            if i < num_folders - 1:
                image_path = os.path.join(pseudomask_dirs[i], file)
            else: 
                image_path = os.path.join(pseudomask_dirs[i], file[:-3]+'tiff')
                
            image = mask2rgb(numpy.array(Image.open(image_path)))
             
            plt.subplot(1, num_folders, i+1)
            plt.imshow(image, aspect='equal')
            plt.axis('off') 
            plt.title(folder_names[i])
                
        # plt.show()
            
        out_path = os.path.join(out_dir,file)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=1)
        plt.close()


if __name__ == '__main__':

    mode_choices = ('val', 'train')

    # ================ parsing arguments ================
    
    parser = argparse.ArgumentParser('Pseudomask Evaluation', add_help=False)
    parser.add_argument('--pseudomask_dir', type=str, required=True,
                        help='Path to pseudomask_dir data.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to trained model.')
    parser.add_argument('--out_dir', type=str, default='.', required=True,
                        help='Path to save logs.')
    parser.add_argument('--cmap_path', type=str, default=None, required=True,
                        help='Path to color map file.')
    parser.add_argument('--mode', type=str, default='val', choices=mode_choices,
                        help='Evaluate on (train, val).')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    main(args.pseudomask_dir, args.data_dir, args.out_dir, args.cmap_path, args.mode)


