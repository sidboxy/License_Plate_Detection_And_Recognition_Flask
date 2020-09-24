import pandas as pd
import numpy as np
import cv2
import os
import pickle
import  time
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from Character_Segmentation import segment_characters

def predict_on_segmentation(model,image_path):
    license_plate = []
    license_plate_number = ''
    model = load_model(model)
    # Detect chars
    digits = segment_characters(image_path)


    for d in digits:
        d = np.reshape(d, (1,28,28,1))
        out = model.predict(d)
        # Get max pre arg
        p = []
        precision = 0
        for i in range(len(out)):
            z = np.zeros(36)
            z[np.argmax(out[i])] = 1.
            precision = max(out[i])
            p.append(z)
        prediction = np.array(p)

        # Inverse one hot encoding
        alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        classes = []
        for a in alphabets:
            classes.append([a])
        ohe = OneHotEncoder()
        ohe.fit(classes)
        pred = ohe.inverse_transform(prediction)
        license_plate.append(pred[0][0])

        if precision > 0:
            print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))

    license_plate_number = ''.join(license_plate)
    print("License Number :" + license_plate_number )
    return license_plate_number