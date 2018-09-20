import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.datasets import cifar10
import time

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    X_train = x_train.astype('float32')
    X_test = x_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    batch_size = 1000
    epochs = 100
    
    cifarInput = Input(shape = (X_train.shape[1:]))
    
    x = Conv2D(32, (3, 3), padding='same')(cifarInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    feature = MaxPooling2D()(x)

    x = Conv2D(8, (3, 3), padding='same')(feature)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)

    decoded = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

    autoencoder = Model(cifarInput, decoded)

    opt = keras.optimizers.adam()

    autoencoder.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])

    start_Time = time.time()

    history = autoencoder.fit(X_train, X_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test, X_test))

    end_Time = time.time() - start_Time
    s = 'batch_size:{0} epoch:{1} time:{2}[sec]\n'.format(batch_size, epochs, end_Time)
    print(s)
    with open('time_log.txt', mode='a') as f:
        f.write(s)

    encoder = Model(cifarInput, feature)

    autoencoder.save('ae_model8.h5')
    encoder.save('ae_encoder8.h5')