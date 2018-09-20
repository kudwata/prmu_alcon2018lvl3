import numpy as np
import matplotlib as plt
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10
import time

import label

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    X_train = x_train.astype('float32')
    X_test = x_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    l_train = label.multilabeling(y_train)
    l_test = label.multilabeling(y_test)

    Y_train0 = keras.utils.to_categorical(l_train[0], 2)
    Y_train1 = keras.utils.to_categorical(l_train[1], 2)
    Y_train2 = keras.utils.to_categorical(l_train[2], 2)
    Y_train3 = keras.utils.to_categorical(l_train[3], 2)
    Y_train4 = keras.utils.to_categorical(l_train[4], 2)
    Y_train5 = keras.utils.to_categorical(l_train[5], 2)
    Y_train6 = keras.utils.to_categorical(l_train[6], 2)
    Y_train7 = keras.utils.to_categorical(l_train[7], 2)
    Y_test0 = keras.utils.to_categorical(l_test[0], 2)
    Y_test1 = keras.utils.to_categorical(l_test[1], 2)
    Y_test2 = keras.utils.to_categorical(l_test[2], 2)
    Y_test3 = keras.utils.to_categorical(l_test[3], 2)
    Y_test4 = keras.utils.to_categorical(l_test[4], 2)
    Y_test5 = keras.utils.to_categorical(l_test[5], 2)
    Y_test6 = keras.utils.to_categorical(l_test[6], 2)
    Y_test7 = keras.utils.to_categorical(l_test[7], 2)

    batch_size = 1000
    epochs = 1000
    
    cifarInput = Input(shape = (X_train.shape[1:]))
    
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(cifarInput)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)

    output0 = Dense(2, activation='softmax', name='output0')(x)
    output1 = Dense(2, activation='softmax', name='output1')(x)
    output2 = Dense(2, activation='softmax', name='output2')(x)
    output3 = Dense(2, activation='softmax', name='output3')(x)
    output4 = Dense(2, activation='softmax', name='output4')(x)
    output5 = Dense(2, activation='softmax', name='output5')(x)
    output6 = Dense(2, activation='softmax', name='output6')(x)
    output7 = Dense(2, activation='softmax', name='output7')(x)

    multiModel = Model(cifarInput, [output0,output1,output2,output3,output4,output5,output6,output7])

    opt = keras.optimizers.adam()

    multiModel.compile(loss={'output0':'categorical_crossentropy','output1':'categorical_crossentropy',
                            'output2':'categorical_crossentropy','output3':'categorical_crossentropy',
                            'output4':'categorical_crossentropy','output5':'categorical_crossentropy',
                            'output6':'categorical_crossentropy','output7':'categorical_crossentropy'},
                            optimizer=opt,metrics=['accuracy'])

    start_Time = time.time()

    es_cb = EarlyStopping(monitor='val_loss',min_delta=0, patience=5, mode='auto')
    tb_cb = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0)

    history = multiModel.fit(X_train,
                            {'output0': Y_train0,
                            'output1': Y_train1,
                            'output2': Y_train2,
                            'output3': Y_train3,
                            'output4': Y_train4,
                            'output5': Y_train5,
                            'output6': Y_train6,
                            'output7': Y_train7},
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test,
                                            {'output0': Y_test0,
                                            'output1': Y_test1,
                                            'output2': Y_test2,
                                            'output3': Y_test3,
                                            'output4': Y_test4,
                                            'output5': Y_test5,
                                            'output6': Y_test6,
                                            'output7': Y_test7}),
                            callbacks=[es_cb, tb_cb])

    end_Time = time.time() - start_Time
    s = 'batch_size:{0} epoch:{1} time:{2}[sec]\n'.format(batch_size, epochs, end_Time)
    print(s)
    with open('time_log.txt', mode='a') as f:
        f.write(s)

    multiModel.save('cnn_model1.h5')