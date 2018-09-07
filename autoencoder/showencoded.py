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

    encoder = load_model('ae_encoder8.h5')

    encoded_imgs = encoder.predict(X_test)

    fig = plt.figure()

    n = 10
    for i in range(1,n+1):
        ax = plt.subplot(1, n, i)
        plt.imshow(encoded_imgs[i].reshape(4, 4*8).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()