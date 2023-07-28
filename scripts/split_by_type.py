import os
import argparse
import matplotlib.pyplot as plt
import csv, numpy
import shutil

def main(in_dir, out_dir, split_ratio, plot_input_stats):
    def plot_stats(type_freq):
        categories = [cat for cat in type_freq.keys()]
        freq = [type_freq[cat] for cat in categories]
        categories = [str(cat) for cat in categories]
        combined_pairs = list(zip(freq, categories))
        sorted_pairs = sorted(combined_pairs, reverse=True)
        freq, categories = zip(*sorted_pairs)
        plt.bar(categories, freq)
        plt.xlabel('Categories')
        plt.ylabel('Frequency')
        plt.show()

    dir_labels = f"{in_dir}/class_labels"

    labels = [file for file in os.listdir(dir_labels) if file.endswith('.csv')]
    labels_path = [os.path.join(dir_labels, file) for file in labels]

    type_freq = {}
    type_images = {}

    for i in range(len(labels_path)):
        with open(labels_path[i], mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            class_one_hot = []
            for row in csv_reader:
                class_one_hot.append(int(row['presence']))
            class_one_hot.insert(0,numpy.sum(class_one_hot))
            class_one_hot = tuple(class_one_hot)
            if class_one_hot not in type_freq.keys():
                type_freq[class_one_hot] = 1
                type_images[class_one_hot] = [labels[i][:-3]]
            else:
                type_freq[class_one_hot] += 1
                type_images[class_one_hot].append(labels[i][:-3])

    if plot_input_stats:
        plot_stats(type_freq)

    for type in type_freq.keys():
        split_num = int(type_freq[type]*split_ratio)
        
        # save train data
        for i in range(0, split_num):
            # copyfile(infile, outfile)
            file = type_images[type][i]
        
            out_path = f"{out_dir}/train/{type[0]}/class_labels"
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(f"{in_dir}/class_labels", file+'csv'), \
                            os.path.join(out_path, file+'csv'))
            
            out_path = f"{out_dir}/train/{type[0]}/images"
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(f"{in_dir}/images", file+'tiff'), \
                            os.path.join(out_path, file+'tiff'))
            
            out_path = f"{out_dir}/train/{type[0]}/masks"
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(f"{in_dir}/masks", file+'tiff'), \
                            os.path.join(out_path, file+'tiff'))
            
        # save val data
        for i in range(split_num, type_freq[type]):
            file = type_images[type][i]

            out_path = f"{out_dir}/val/{type[0]}/class_labels"
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(f"{in_dir}/class_labels", file+'csv'), \
                            os.path.join(out_path, file+'csv'))
            
            out_path = f"{out_dir}/val/{type[0]}/images"
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(f"{in_dir}/images", file+'tiff'), \
                            os.path.join(out_path, file+'tiff'))
            
            out_path = f"{out_dir}/val/{type[0]}/masks"
            os.makedirs(out_path, exist_ok=True)
            shutil.copyfile(os.path.join(f"{in_dir}/masks", file+'tiff'), \
                            os.path.join(out_path, file+'tiff'))
            
if __name__ == '__main__':

    mode_choices = ('val', 'train')

    # ================ parsing arguments ================
    
    parser = argparse.ArgumentParser('Split by Image Type', add_help=False)
    parser.add_argument('--in_dir', type=str, required=True,
                        help='Path to input data.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to output directory.')
    parser.add_argument('--split_ratio', type=float, required=True,
                        help='Train-Val split ratio.')
    parser.add_argument('--plot_stats', type=bool, default=False,
                        choices=(True, False), help='Plot stats of input data.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(args.plot_stats)
    main(args.in_dir, args.out_dir, args.split_ratio, args.plot_stats)
