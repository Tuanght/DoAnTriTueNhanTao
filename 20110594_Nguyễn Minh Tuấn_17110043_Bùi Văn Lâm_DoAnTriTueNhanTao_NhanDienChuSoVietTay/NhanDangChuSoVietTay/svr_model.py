from  flask import Flask, redirect, url_for, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2

#load model
model_architecture = "digit_config.json"
model_weights = "digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 

#train
optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]) 

# Loading MNIST dataset.
# verify
# You can verify that the split between train and test is 60,000, and 10,000 respectively. 
# Labels have one-hot representation.is automatically applied
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test_image = X_test

# X_train is 60000 rows of 28x28 values; we --> reshape it to 
# 60000 x 784
RESHAPED = 784
#
X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')

# Normalize inputs to be within in [0,1]
X_test /= 255

# Variables global
index = np.random.randint(0, 9999, 150)
count = 0

app= Flask(__name__)

@app.route('/', methods=['GET','POST'])
def trang_chu():
    global index, count
    if request.method == 'POST':
        index = np.random.randint(0, 9999, 150)
        count = 1
        digit_random = np.zeros((10*28, 15*28), dtype = np.uint8)
        for i in range(0, 150):
            m = i // 15
            n = i % 15
            digit_random[m*28:(m+1)*28, n*28:(n+1)*28] = X_test_image[index[i]] 
        cv2.imwrite('static\image\digit_random.jpg', digit_random)
        return render_template('index.html',src_image=url_for('static',filename='image/digit_random.jpg'))

    if request.method == 'GET':
        if count == 1:
            X_test_sample = np.zeros((150,784), dtype = np.float32)
            for i in range(0, 150):
                X_test_sample[i] = X_test[index[i]]

            prediction = model.predict(X_test_sample)
            s = '   '
            for i in range(0, 150):
                ket_qua = np.argmax(prediction[i])
                s = s + str(ket_qua) + '      '
                if (i+1) % 15 == 0:
                    s = s + '\n   '
            return render_template('index.html',value_predict=s,src_image=url_for('static',filename='image/digit_random.jpg'))
            
    return render_template('index.html',src_image=url_for('static',filename='image/black-image.jpg'))

if __name__=="__main__":
    app.run(debug=True)