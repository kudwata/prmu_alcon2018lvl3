import numpy as np
import sys
import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from glob import glob
import os
import time

import clone

# ラベルリスト
LT = clone.LT

# クローン認識器訓練用画像が存在するディレクトリのパス
TRAIN_IMAGE_DIR = clone.TRAIN_IMAGE_DIR
# 訓練用画像の総数
TRAIN_IMAGE_NUM = 418522
# 学習に使う画像の数
N_TRAIN = 10000
# テストデータの割合
VAL_R = 0.1

# 中間層のユニット数
UNIT_NUM = 128

def getNewestModel(model, dirname):
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model

if __name__ == "__main__":
    train_set = clone.LV3_ImageSet(TRAIN_IMAGE_DIR)

    train_index = range(N_TRAIN)
    #train_index = random.sample(range(TRAIN_IMAGE_NUM), N_TRAIN)

    x_train = train_set.get_image_list(train_index)
    X_train = x_train.astype('float32')
    X_train /= 255.0

    extractor = clone.LV3_FeatureExtractor()

    target_dir = "../lv3_targets/classifier_01/"
    target = clone.LV3_TargetClassifier()
    target.load(target_dir + "train.csv")

    features = train_set.get_feature_list(train_index, extractor)

    likelihoods = target.predict_proba(features)

    Y_train = []
    for i in range(LT.N_LABELS()):
        y_train = []
        for j in range(N_TRAIN):
            y_train.append([likelihoods[j][i], 1 - likelihoods[j][i]])
        Y_train.append(np.array(y_train, dtype='float32'))
    
    input_layer = Input(shape = (X_train.shape[1:]))
    x = Conv2D(UNIT_NUM, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(UNIT_NUM, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(UNIT_NUM, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(UNIT_NUM, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    feature = Dense(512, activation='relu')(x)
    feature = Dropout(0.25)(feature)

    output = []
    for i in range(LT.N_LABELS()):
        output_layer = Dense(64, activation='relu')(feature)
        output_layer = Dense(2, activation='softmax', name=LT.ID2LNAME(i))(output_layer)
        output.append(output_layer)

    model = Model(input_layer, output)

    opt = keras.optimizers.adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    
    #es_cb = EarlyStopping(monitor='val_loss',min_delta=0, patience=5, mode='auto')
    tb_cb = TensorBoard(log_dir='logs', histogram_freq=0)
    mc_cb = ModelCheckpoint('weight/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, save_weights_only=True)

    callback = [tb_cb, mc_cb]

    batch_size = 50
    epochs = 200
    
    start_Time = time.time()
    
    model.fit(x=X_train,
            y=Y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callback,
            validation_split=0.1)

    end_Time = time.time() - start_Time
    s = 'batch_size:{0} epoch:{1} time:{2}[sec]\n'.format(batch_size, epochs, end_Time)
    print(s)
    with open('time_log.txt', mode='a') as f:
        f.write(s)

    model = getNewestModel(model, "weight/")
    model.save("cnn_model.h5")