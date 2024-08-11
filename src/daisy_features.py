from skimage.feature import daisy
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np


def apply_daisy(row, shape):
    feat = daisy(row.reshape(shape), step=100, radius=35, rings=3,
                 histograms=8, orientations=4, visualize=False)
    return feat.reshape((-1))


def extract_daisy_features(images, h=512, w=512):
    daisy_features = np.apply_along_axis(apply_daisy, 1, images, (h, w))
    return daisy_features


def visualize_daisy_descriptor(images, index, h=512, w=512):
    img = images[index].reshape((h, w))
    features, img_des = daisy(
        img, step=100, radius=35, rings=3, histograms=8, orientations=4, visualize=True)
    imshow(img_des)
    plt.show()
