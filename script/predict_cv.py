import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import alpha_blend
    
def predict(image):
    ori_image = image.copy()
    h,w,ch = image.shape
    prediction = np.zeros(shape=(h,w))

    # print(np.average(image))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[:80,:] = 125
    # image = np.clip(image*3,0,255)
    image = cv2.equalizeHist(image)
    ret, image = cv2.threshold(image, 27, 255, cv2.THRESH_BINARY_INV)#27

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,kernel,iterations=1)
    image = cv2.morphologyEx(image,cv2.MORPH_DILATE,kernel,iterations=3)
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,kernel,iterations=2)

    # cv2.imshow('threshold',image)

    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # stats[:,x] = [x, y, width, height, area]
    sizes = stats[:, -1]

    # for S1
    # print(stats)
    # for i in range(nb_components):
    #     if 2500 < sizes[i] < 12000:
    #         if 55 < stats[i,2] < max(120,sizes[i]/70):
    #             if 55 < stats[i,3] < max(120,sizes[i]/70):
    #                 # print(stats[i])
    #                 prediction[output == i] = 255
    #                 break

    # for S4
    # print(stats)
    for i in range(nb_components):
        if 400 < sizes[i] < 12000:
            if np.sqrt(sizes[i])*0.7 < stats[i,2] < np.sqrt(sizes[i])*1.5:
                if np.sqrt(sizes[i])*0.7 < stats[i,3] < np.sqrt(sizes[i])*1.5:
                    # print(stats[i])
                    prediction[output == i] = 255
                    break
            



    if np.sum(prediction.flatten()) > 0:
        confidence = 1
    else:
        confidence = 0

    return prediction, confidence


if __name__ == '__main__':
    dataset_path = r'..\dataset\public\S4\23'
    nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
    print('number of image:',nr_image)
    image = cv2.imread(os.path.join(dataset_path, '0.jpg'))
    h,w,ch = image.shape


    dpi = matplotlib.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w / dpi, h / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    for idx in range(nr_image):
        image = cv2.imread(os.path.join(dataset_path, f'{idx}.jpg'))
        label = cv2.imread(os.path.join(dataset_path, f'{idx}.png'))

        prediction, confidence = predict(image)

        blended  = alpha_blend(image, label)
        blended2 = alpha_blend(image,np.stack((prediction,) * 3, axis=-1))
        ax.clear()
        ax.imshow(blended2)
        ax.axis('off')
        plt.draw()
        plt.pause(0.01)
    plt.close()


    idx = 9
    # for idx in range(100):
    print(idx)
    image = cv2.imread(os.path.join(dataset_path, f'{idx}.jpg'))
    label = cv2.imread(os.path.join(dataset_path, f'{idx}.png'), 0)
    prediction, confidence = predict(image)
    cv2.imshow('GT',label)
    
    cv2.imshow('prediction',alpha_blend(image,np.stack((prediction,) * 3, axis=-1)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()