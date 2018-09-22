import numpy as np
import sys
import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import clone

# ラベルリスト
LT = clone.LT

# クローン認識器訓練用画像が存在するディレクトリのパス
TRAIN_IMAGE_DIR = clone.TRAIN_IMAGE_DIR
# 訓練用画像の総数
TRAIN_IMAGE_NUM = 418522
# 学習に使う画像の数
N_TRAIN = 100000
# テストデータの割合
VAL_R = 0.1

# 中間層のユニット数
UNIT_NUM = 128

if __name__ == "__main__":
    train_set = clone.LV3_ImageSet(TRAIN_IMAGE_DIR)

    train_index = range(N_TRAIN)
    #train_index = random.sample(range(TRAIN_IMAGE_NUM), N_TRAIN)

    x_train = train_set.get_image_list(train_index)
    X_train = x_train.astype('float32')
    X_train /= 255.0
    
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
        output_layer = Dense(2, Activation='softmax', name=LT.ID2LNAME(i))(feature)
        output.append(output_layer)

    model = Model(input_layer, output)
    
    es_cb = EarlyStopping(monitor='val_loss',min_delta=0, patience=5, mode='auto')
    tb_cb = TensorBoard(log_dir='logs', histogram_freq=0)
    mc_cb = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)

    callback = [es_cb, tb_cb, mc_cb]

    batch_size = 1000
    epochs = 1000