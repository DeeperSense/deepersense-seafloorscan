#!/usr/bin/env python3

if __name__ == '__main__':

    import os
    import argparse
    import csv
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser('Split Data by Number of Classes per Image', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to output directory.')
    parser.add_argument('--num_classes', type=int, default=0,
                        help='Number of classes.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.data_dir) and os.path.isdir(args.data_dir):
        # Get a list of all .tiff files in the input folder
        input_files = [f for f in os.listdir(args.data_dir) if f.endswith('.tiff')]

        # Loop through each file in the input folder
        for file_name in input_files:
            # Open the mask image using PIL
            img = Image.open(os.path.join(args.data_dir, file_name))

            # Convert the image to a numpy array
            img_array = np.array(img)

            # Find the unique values in the array
            classes = np.sort(np.unique(img_array))

            # Create a dictionary to store the class labels and their presence in the image
            class_dict = {c: 0 for c in range(args.num_classes)}

            # Loop through each class and check if it is present in the image
            for c in classes:
                if c in class_dict:
                    class_dict[c] = 1

            # Create the output CSV file with the same name as the input file
            output_file = os.path.join(args.out_dir, os.path.splitext(file_name)[0] + '.csv')

            # Write the class labels and their presence to the CSV file
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['class', 'presence'])
                for k, v in class_dict.items():
                    writer.writerow([k, v])
    else:
        raise OSError('Not a valid input directory!')
