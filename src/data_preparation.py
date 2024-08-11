import os
import random
import numpy as np
from PIL import Image


def load_and_resize_image(base_folder, target_size, num):
    images, labels = [], []
    categories = os.listdir(base_folder)

    for category in categories:
        category_path = os.path.join(base_folder, category)
        if not os.path.isdir(category_path):
            continue
        # randomly select images
        filenames = os.listdir(category_path)
        random.shuffle(filenames)
        for filename in filenames[:num // len(categories)]:
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                img_path = os.path.join(category_path, filename)
                img = Image.open(img_path)
                img_gray = img.convert('L')
                img_resized = img_gray.resize(target_size, Image.LANCZOS)
                img_array = np.array(img_resized).flatten()
                images.append(img_array)
                labels.append(category)
    return np.array(images), np.array(labels)


def get_data(folder_path, target_size, num_images):
    images, labels = load_and_resize_image(
        folder_path, target_size, num_images)
    return images, labels
