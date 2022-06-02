import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import alpha_blend
from predict import predict,generate_output_file



def predict_cv(image):
    ori_image = image.copy()
    h,w,ch = image.shape
    prediction = np.zeros(shape=(h,w))

    # print(np.average(image))

    # cv2.imshow('ori_image',ori_image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[:80,:] = 125

    image = cv2.equalizeHist(image)

    cv2.imshow('equalizeHist',image)

    # image = cv2.medianBlur(image,9)
    # cv2.imshow('median_blur',image)

    canny = cv2.Canny(image, 15, 150)
    cv2.imshow('Canny',canny)

    # blur = cv2.blur(image, (5, 5))

    # laplacian = cv2.Laplacian(blur,cv2.CV_64F)
    # cv2.imshow('laplacian',laplacian)

    ret, image = cv2.threshold(image, 27, 255, cv2.THRESH_BINARY_INV)#27

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,kernel,iterations=1)
    image = cv2.morphologyEx(image,cv2.MORPH_DILATE,kernel,iterations=3)
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,kernel,iterations=2)

    # cv2.imshow('threshold',image)

    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # stats[:,x] = [x, y, width, height, area]
    sizes = stats[:, -1]

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


# resnet18.pth:
# OK : 26 04 05 06 08 09 10 11 12 13
# NO : 01 02 03 07 

if __name__ == '__main__':
    dataset_path = r'..\dataset\public\S5\13'
    nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
    print('number of image:',nr_image)
    image = cv2.imread(os.path.join(dataset_path, '0.jpg'))
    h,w,ch = image.shape


    dpi = matplotlib.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w / dpi, h / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    for idx in range(nr_image):
        image = cv2.imread(os.path.join(dataset_path, f'{idx}.jpg'))

        prediction, confidence = predict(image)
        # prediction = 255*prediction
        # prediction = reserve_largest_component(prediction)

        # prediction, confidence = predict_cv(image)



        blended2 = alpha_blend(image,np.stack((prediction,) * 3, axis=-1))

        # blended2 = prediction
        ax.clear()
        ax.imshow(blended2)
        ax.axis('off')
        plt.draw()
        plt.pause(0.01)
    plt.close()


    # idx = 9
    # # for idx in range(100):
    # print(idx)
    # image = cv2.imread(os.path.join(dataset_path, f'{idx}.jpg'))
    # label = cv2.imread(os.path.join(dataset_path, f'{idx}.png'), 0)
    # prediction, confidence = predict(image)
    # cv2.imshow('GT',label)
    
    # cv2.imshow('prediction',alpha_blend(image,np.stack((prediction,) * 3, axis=-1)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()