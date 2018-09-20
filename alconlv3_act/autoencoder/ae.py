import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import random

import clone

# クローン認識器訓練用画像が存在するディレクトリのパス
TRAIN_IMAGE_DIR = clone.TRAIN_IMAGE_DIR
# クローン認識器評価用画像が存在するディレクトリのパス
VALID_IMAGE_DIR = clone.VALID_IMAGE_DIR
# 訓練用画像の総数
TRAIN_IMAGE_NUM = 418522
# 評価用画像の総数
VALID_IMAGE_NUM = 104630
#学習に使う画像の数
N_TRAIN = 10000
#テストに使う画像の数
N_TEST = 1000

if __name__ == '__main__':
    train_set = clone.LV3_ImageSet(TRAIN_IMAGE_DIR)
    test_set = clone.LV3_ImageSet(VALID_IMAGE_DIR)

    train_index = range(N_TRAIN)
    #train_index = random.sample(range(TRAIN_IMAGE_NUM), N_TRAIN)
    test_index = range(N_TEST)
    #test_index = random.sample(range(VALID_IMAGE_NUM), N_TEST)

    x_train = train_set.get_image_list(train_index)

    x_test = test_set.get_image_list(test_index)
    
    X_train = x_train.astype('float32')
    X_test = x_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    batch_size = 100
    epochs = 100
    
    cifarInput = Input(shape = (X_train.shape[1:]))
    
    x = Conv2D(32, (3, 3), padding='same')(cifarInput)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(4, (3, 3), padding='same')(x)
    x = Conv2D(4, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    feature = MaxPooling2D()(x)

    x = Conv2D(4, (3, 3), padding='same')(feature)
    x = Conv2D(4, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)

    decoded = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

    autoencoder = Model(cifarInput, decoded)

    opt = keras.optimizers.adam()

    autoencoder.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])

    start_Time = time.time()
    
    es_cb = EarlyStopping(monitor='val_loss',min_delta=0, patience=3, mode='auto')
    tb_cb = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0)

    history = autoencoder.fit(X_train, X_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test, X_test),
                            callbacks=[es_cb, tb_cb])

    end_Time = time.time() - start_Time
    s = 'batch_size:{0} epoch:{1} time:{2}[sec]\n'.format(batch_size, epochs, end_Time)
    print(s)
    with open('time_log.txt', mode='a') as f:
        f.write(s)

    encoder = Model(cifarInput, feature)

    autoencoder.save('ae_model.h5')
    encoder.save('ae_encoder.h5')