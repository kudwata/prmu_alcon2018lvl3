import sys
import clone
import numpy as np

TRAIN_IMAGE_DIR = clone.TRAIN_IMAGE_DIR

train_set = clone.LV3_ImageSet(TRAIN_IMAGE_DIR)

extractor = clone.LV3_FeatureExtractor()
LT = clone.LT

target_dir = sys.argv[1]
target = clone.LV3_TargetClassifier()
target.load(target_dir + "train.csv")

n = 20
features = train_set.get_feature_list(range(n), extractor)

likelihoods = target.predict_proba(features)
print(likelihoods)

Y_train = []
for i in range(LT.N_LABELS()):
    y_train = []
    for j in range(n):
        y_train.append([likelihoods[j][i], 1 - likelihoods[j][i]])
    Y_train.append(np.array(y_train, dtype='float32'))

print(Y_train)