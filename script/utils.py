import numpy as np
from PIL import Image


def alpha_blend(input_image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5):
    """Alpha Blending utility to overlay segmentation masks on input images
    Args:
        input_image: a np.ndarray with 1 or 3 channels
        segmentation_mask: a np.ndarray with 3 channels
        alpha: a float value
    """
    if len(input_image.shape) == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    blended = input_image.astype(np.float32) * alpha + segmentation_mask.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    return blended


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count

if __name__ == '__main__':
    import os
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt
    # Object for prediction
    from predict import predict
    use_label = False
    dataset_path = "../dataset/public/S5/01"
    nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
    print(nr_image)
    image = np.asarray(Image.open(os.path.join(dataset_path, '0.jpg')))
    h = image.shape[0]
    w = image.shape[1]
    dpi = matplotlib.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w / dpi, h / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    for idx in range(nr_image):
        image_name = os.path.join(dataset_path, f'{idx}.jpg')
        label_name = os.path.join(dataset_path, f'{idx}.png')
        image = np.asarray(Image.open(image_name))
        if use_label:
            label = np.asarray(Image.open(label_name))
        else:
            label, _ = predict(image)
            zeros = np.zeros_like(label)
            label = np.stack((label, zeros, label), axis=-1) * 255
        blended = alpha_blend(image, label, 0.5)
        ax.clear()
        ax.imshow(blended)
        ax.axis('off')
        plt.draw()
        plt.pause(0.01)
    plt.close()
