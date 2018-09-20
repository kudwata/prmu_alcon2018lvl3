import numpy as np
import keras
from keras.datasets import cifar10
import label

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    l_test = label.multilabeling(y_test)

    for i in range(8):
        labels = keras.utils.to_categorical(l_test[i], 2)
        np.save('output/model0_{}.npy'.format(i), labels)