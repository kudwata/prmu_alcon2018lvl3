import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import load_model

from sklearn.cluster import KMeans
from sklearn import svm

model_num = 1
labels = []
for i in range(8):
    labels.append(np.load('model{0}/model{0}_{1}.npy'.format(model_num, i)))

def clusterring(images, N_CLUSTERS=64):
    cluster = KMeans(n_clusters=N_CLUSTERS)
    predict = cluster.fit_predict(images)
    center = cluster.cluster_centers_

    return predict, center

def getLabel(number):
    ans = np.zeros(8)
    for i in range(8):
        ans[i] = np.argmax(labels[i][number])

    return ans

def clone(N_CLUSTERS=32):
    (_, _), (x_test, _) = cifar10.load_data()
    
    X_test = x_test.astype('float32')
    X_test /= 255.0

    encoder = load_model('ae_encoder8.h5')

    encoded_imgs = encoder.predict(X_test)
    feature = encoded_imgs.reshape(10000, 128)

    n_clusters = N_CLUSTERS
    
    clusters, center = clusterring(feature, N_CLUSTERS=n_clusters)

    near_num = np.zeros(n_clusters, dtype=int)
    near_far = np.full(n_clusters, 100)
    for j in range(10000):
        far = np.linalg.norm(feature[j] - center[clusters[j]])
        if near_far[clusters[j]] > far:
            near_far[clusters[j]] = far
            near_num[clusters[j]] = j

    predicted = np.zeros((8, n_clusters))
    p_features = np.zeros((n_clusters, 128))
    for i in range(n_clusters):
        for j in range(8):
            predicted[j][i] = getLabel(near_num[i])[j]
        p_features[i] = feature[near_num[i]]

    expected = np.zeros((10000,8))

    for i in range(8):
        clf = svm.SVC()
        in_arr = predicted[i]
        clf.fit(p_features, in_arr)
        
        temp = clf.predict(feature)

        for j in range(10000):
            expected[j][i] = temp[j]

    np.save('target_svm/model{0}/output{1}.npy'.format(model_num, n_clusters), expected)

if __name__ == '__main__':
    samplings = np.array((16,32,64,128,256,512,1024,2048,3072,4096,6144,8192,10000))
    for i in samplings:
        clone(i)