import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model

import clone
# クローン認識器訓練用画像が存在するディレクトリのパス
TRAIN_IMAGE_DIR = clone.TRAIN_IMAGE_DIR
# クローン認識器評価用画像が存在するディレクトリのパス
VALID_IMAGE_DIR = clone.VALID_IMAGE_DIR

if __name__ == '__main__':
    n = 10

    test_set = clone.LV3_ImageSet(VALID_IMAGE_DIR)

    test_index = range(n+1)
    #test_index = random.sample(range(VALID_IMAGE_NUM), n+1)

    x_test = test_set.get_image_list(test_index)
    
    X_test = x_test.astype('float32')
    X_test /= 255.0

    decoder = load_model('ae_model.h5')

    decoded_imgs = decoder.predict(X_test)

    fig = plt.figure(figsize=(8,6), dpi=200)

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