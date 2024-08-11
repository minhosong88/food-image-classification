import matplotlib.pyplot as plt
import random

# Helper Plotting Function


def visualize_image(images, labels, h=512, w=512, n_row=3, n_col=6):
    indices = list(range(len(images)))
    random.shuffle(indices)

    plt.figure(figsize=(1.7 * n_col, 2.3 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        i = indices[i]
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(labels[i], size=12)
        plt.xticks(())
        plt.yticks(())
