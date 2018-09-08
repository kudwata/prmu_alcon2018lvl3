import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import load_model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    X_test = x_test.astype('float32')
    X_test /= 255.0

    model = load_model('lccnn_model1.h5')

    labels = model.predict(X_test)

    for i in range(8):
        np.save('output/model3_{}.npy'.format(i), labels[i])