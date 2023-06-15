#!/usr/bin/env python3

if __name__ == '__main__':
    import os
    import sys
    import shutil
    from datetime import datetime
    from sklearn.model_selection import train_test_split

    try:
        data_dir = sys.argv[1]
        test_size = int(sys.argv[2])
    except:
        print('Usage: train_test_split <dataset_dir> <test_data_size>')
    else:
        if os.path.exists(data_dir):
            os.chdir(data_dir)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

            train_images_dir = os.path.join('train_' + timestamp, 'images')
            os.makedirs(train_images_dir)
            train_masks_dir = os.path.join('train_' + timestamp, 'masks')
            os.makedirs(train_masks_dir)
            test_images_dir = os.path.join('test_' + timestamp, 'images')
            os.makedirs(test_images_dir)
            test_masks_dir = os.path.join('test_' + timestamp, 'masks')
            os.makedirs(test_masks_dir)

            train, test = train_test_split(os.listdir('images'), test_size=test_size)

            for file_name in train:
                shutil.copy(os.path.join('images', file_name), train_images_dir)
                shutil.copy(os.path.join('masks', file_name), train_masks_dir)
            for file_name in test:
                shutil.copy(os.path.join('images', file_name), test_images_dir)
                shutil.copy(os.path.join('masks', file_name), test_masks_dir)
        else:
            sys.exit('Invalid Path!')
