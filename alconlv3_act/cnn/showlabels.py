import numpy as np
import sys
import matplotlib.pyplot as plt
#import keras
#from keras.models import Model
#from keras.models import load_model

import clone

# ラベルリスト
LT = clone.LT

# pylint: disable=E1136

if __name__ == "__main__":
#    model = load_model("cnn_model.h5")

    if len(sys.argv) < 3:
        print("usage: clone.py /image_dir /target/classifier/path")
        exit(0)

    img_dir = sys.argv[1]
    target_dir = sys.argv[2]

    img_set = clone.LV3_ImageSet(img_dir)
    
    target = clone.LV3_TargetClassifier()
    target.load(target_dir + "train.csv")

    while(True):
        s = input("img_num:")
        if s == "":
            exit()
        img_num = int(s)
        labels = target.predict_once([img_num, 0])
        for i in range(LT.N_LABELS()):
            likelihood = labels[i]
            if likelihood >= 0.9:
                l_name = LT.ID2LNAME(i)
                print("{0}:{1}".format(l_name, likelihood))
        img = img_set.get_image(img_num)
        img = img.astype('float32')
        img /= 255.0
        plt.imshow(img)
        plt.axis("off")
        plt.show()