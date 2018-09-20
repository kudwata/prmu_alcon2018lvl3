import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import load_model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    X_train = x_train.astype('float32')
    X_test = x_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    decoder = load_model('ae_model8.h5')

    decoded_imgs = decoder.predict(X_test)

    fig = plt.figure()

    n = 10
    for i in range(1,n+1):
        ax = plt.subplot(2, n, i)
        plt.imshow(X_test[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i+n)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()