import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import cv2


def true_white(image):
    image_new = []
    for pixel in image[0]:
        if pixel >= 100:
            pixel = 255
        image_new += [pixel]
    for n in range(784):
        if n % 28 == 0 or n % 28 == 1 or n % 28 == 27 or n % 28 == 26:
            image_new[n] = 255
    return image_new


def non_white(image):
    pixel_list = []
    for n in range(28):
        for m in range(28):
            if image[n][m] != 255:
                pixel_list += [[n, m, image[n][m]]]
    return pixel_list


def hor_ver_cont(image, zero=False):
    pixel_list = non_white(image)
    for i in range(len(pixel_list)):
        for j in range(len(pixel_list)):
            if pixel_list[i][0] == pixel_list[j][0] + 1 and pixel_list[i][1] == pixel_list[j][1]:
                pixel_list += [[pixel_list[i][0] + 2, pixel_list[j][1], pixel_list[i][2] * (zero == False)]]
                pixel_list += [[pixel_list[i][0] - 2, pixel_list[j][1], pixel_list[i][2] * (zero == False)]]
            if pixel_list[i][1] == pixel_list[j][1] + 1 and pixel_list[i][0] == pixel_list[j][0]:
                pixel_list += [[pixel_list[i][0], pixel_list[j][1] - 2, pixel_list[i][2] * (zero == False)]]
                pixel_list += [[pixel_list[i][0], pixel_list[j][1] + 2, pixel_list[i][2] * (zero == False)]]
    return pixel_list


def adj_cont(image, zero=False):
    pixel_list = non_white(image)
    for n in range(28):
        for m in range(28):
            for i in range(len(pixel_list)):
                if pixel_list[i][0] == n and pixel_list[i][1] == m and n != 0 and n != 27 and m != 0 and m != 27:
                    image[n + 1][m] = pixel_list[i][2] * (zero == False)
                    image[n - 1][m] = pixel_list[i][2] * (zero == False)
                    image[n][m + 1] = pixel_list[i][2] * (zero == False)
                    image[n][m - 1] = pixel_list[i][2] * (zero == False)
    return image


def check_img(image):
    plt.imshow(image.reshape((28, 28)))
    plt.show()


def contrast_change(image, ratio):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(ratio)
    return image


def color_reverse(image):
    for i in range(len(image[0])):
        image[0][i] = 255 - image[0][i]
    return image


model = keras.models.load_model('mnist 9.model')
videoCaptureObject = cv2.VideoCapture(0)
result = True

while result:
    ret, img = videoCaptureObject.read()
    resize = cv2.resize(img, (28, 28))
    cv2.imwrite("C:/Users/Val/Desktop/mnist/sample pic.jpg", resize)
    img = Image.open("C:/Users/Val/Desktop/mnist/sample pic.jpg")

    img = contrast_change(img, 1.5)
    img = ImageOps.grayscale(img)
    img = img.getdata()
    img = np.array([img])
    img_new = true_white(img)
    img = np.array(img_new).reshape((28, 28))
    check_img(img)

    PIXEL_LIST = hor_ver_cont(img, zero=True)
    img = adj_cont(img, zero=True)
    img = img.reshape((-1, 28 * 28))
    img = color_reverse(img)
    check_img(img)

    img = tf.keras.utils.normalize(img, axis=1)
    prediction = model.predict(img)
    prediction = np.argmax(prediction)
    print(prediction)

    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()
