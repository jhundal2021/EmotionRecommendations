import os
import cv2
import sys
import numpy as np
import random
import tensorflow as tf
from time import sleep
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from tensorflow.keras.optimizers import RMSprop,Adam,SGD

"""config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.75))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)"""


key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
sleep(2)
while True:

    try:
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        print(frame)  # prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='input/saved_img.jpg', img=frame)
            webcam.release()
            break

        elif key == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

imagePath = 'input/saved_img.jpg'
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(48, 48)
)

path = ""
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 1)
    roi_color = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    print("[INFO] Object found. Saving locally.")
    roi_color = cv2.resize(gray, (48, 48))
    cv2.imwrite('input/' + str(w) + str(h) + '_faces.jpg', roi_color)
    path = 'input/' + str(w) + str(h) + '_faces.jpg'


model = tf.keras.models.load_model('models/faceFit')
photo = tf.keras.preprocessing.image.load_img(path, target_size=(48,48))
data = keras.preprocessing.image.img_to_array(photo)
data = np.array([data])

result = model.predict(data)
print(result)

max_ind = np.where(result == np.amax(result))

print(max_ind)

