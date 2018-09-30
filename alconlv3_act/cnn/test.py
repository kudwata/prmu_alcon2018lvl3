import sys
import clone
import numpy as np
import matplotlib.pyplot as plt

TRAIN_IMAGE_DIR = clone.TRAIN_IMAGE_DIR

train_set = clone.LV3_ImageSet(TRAIN_IMAGE_DIR)

extractor = clone.LV3_FeatureExtractor()
LT = clone.LT

target_dir = sys.argv[1]
target = clone.LV3_TargetClassifier()
target.load(target_dir + "train.csv")

n = 1000
features = train_set.get_feature_list(range(n), extractor)

likelihoods = target.predict_proba(features)

hist, bins = np.histogram(likelihoods, bins=10, range=(0,1), density=True)

plt.hist(likelihoods.reshape((-1)))
plt.show()