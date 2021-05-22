from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
import numpy as np
import joblib
from PIL import Image, ImageOps
import pandas as pd
import cv2

x_train = pd.read_csv('C:/Users/Val/Desktop/mnist/train.csv')
y_train = x_train.pop('label')
x_train = x_train.values
x_test = pd.read_csv('C:/Users/Val/Desktop/mnist/test.csv')
x_test = x_test.values
x_train = x_train/256
x_test = x_test/256

clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64))
clf.fit(x_train, y_train)
# joblib.dump(clf, 'clf.pkl')
# clf = joblib.load('clf.pkl')

videoCaptureObject = cv2.VideoCapture(0)
result = True
while(result):
    ret, img = videoCaptureObject.read()
    # img = ImageOps.grayscale(img)
    resize = cv2.resize(img, (28, 28))
    cv2.imwrite("C:/Users/Val/Desktop/mnist/sample pic.jpg", resize)
    img = Image.open("C:/Users/Val/Desktop/mnist/sample pic.jpg")
    # img.show()
    img = ImageOps.grayscale(img)
    # img.show()

    img = img.getdata()
    # print(img)
    img = np.array(img)
    # print(img)
    img = img.reshape((-1, 28 * 28))
    # print(img)
    img = scale(img)
    # print(img)
    prediction = clf.predict(img)
    print(prediction)
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()

