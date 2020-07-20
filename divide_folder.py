# This python file is for spliting the image folder into train/test/valid set

import os, os.path, shutil

data_root = 'data/'

train_files = []
with open(data_root + 'Flickr_8k.trainImages.txt', 'r') as reader:
    for line in reader:
        train_files.append(line[:-1]) # eliminate '\n'

test_files = []
with open(data_root + 'Flickr_8k.testImages.txt', 'r') as reader:
    for line in reader:
        test_files.append(line[:-1]) # eliminate '\n'

validation_files = []
with open(data_root + 'Flickr_8k.devImages.txt', 'r') as reader:
    for line in reader:
        validation_files.append(line[:-1]) # eliminate '\n'

folder_path = "data/Flicker8k_Dataset"
images = [f for f in os.listdir(folder_path)]

for image in images:
    if image in train_files:
        new_path = 'data/train/train'
    elif image in test_files:
        new_path = 'data/test/test'
    elif image in validation_files:
        new_path = 'data/valid/valid'
    else:
        continue

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    old_image_path = os.path.join(folder_path, image)
    new_image_path = os.path.join(new_path, image)
    shutil.move(old_image_path, new_image_path)