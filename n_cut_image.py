from skimage import data, segmentation, color
from skimage.exposure import histogram
import matplotlib.pyplot as plt
from skimage.future import graph
import numpy as np


def create_mask(filename, n_segments=400, n_cuts=10):
    img = data.load(filename)

    labels1 = segmentation.slic(img, n_segments=n_segments)

    rag = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, rag, num_cuts=n_cuts)

    return labels2, img

def save_segments(labels, img, image_name, mask=-1):
    for value in np.unique(labels):
        temp_label = np.where(labels == value, value, mask)
        # This creates a white-black image
        temp_out = color.label2rgb(temp_label, img, bg_label=mask, colors=['white', 'black'], image_alpha=0, alpha=1)

        # This overlays on the average of background
        # temp_out = color.label2rgb(temp_label, img, kind='avg')

        plt.imshow(temp_out)
        plt.axis('off')
        plt.savefig('vocabulary/' + image_name + '_mask_{}'.format(value) + '.jpg')


if __name__ == "__main__":
    file = '/Pictures/2_1279.jpg'
    image_name = file.split('/')[10].split('.')[0]

    labels, img = create_mask(file)

    save_segments(labels, img, image_name)



