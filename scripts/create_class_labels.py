#!/usr/bin/env python3

if __name__ == '__main__':
    import os
    import sys
    import csv
    import numpy as np
    from PIL import Image
    try:
        no_of_class = int(sys.argv[1])
        input_path = sys.argv[2]
        output_path = sys.argv[3]
    except:
        print('Usage: create_class_label <no_of_classes> <input_dataset_dir> <output_dataset_dir>')
    else:
        # Get a list of all .tiff files in the input folder
        input_files = [f for f in os.listdir(input_path) if f.endswith('.tiff')]

        # Loop through each file in the input folder
        for file_name in input_files:
            # Open the mask image using PIL
            img = Image.open(os.path.join(input_path, file_name))

            # Convert the image to a numpy array
            img_array = np.array(img)

            # Find the unique values in the array
            classes = np.sort(np.unique(img_array))

            # Create a dictionary to store the class labels and their presence in the image
            class_dict = {c: 0 for c in range(no_of_class)}

            # Loop through each class and check if it is present in the image
            for c in classes:
                if c in class_dict:
                    class_dict[c] = 1

            # Create the output CSV file with the same name as the input file
            output_file = os.path.join(output_path, os.path.splitext(file_name)[0] + '.csv')

            # Write the class labels and their presence to the CSV file
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['class_label', 'presence'])
                for k, v in class_dict.items():
                    writer.writerow([k, v])
