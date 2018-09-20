import numpy as np

def multilabeling(d_label):
    label = np.zeros((8,d_label.shape[0]))
    for i in range(d_label.shape[0]):
        if d_label[i][0] in [0,1,8,9]:
            label[0][i] = 1
        if d_label[i][0] in [2,3,4,5,6,7]:
            label[1][i] = 1
        if d_label[i][0] in [0,2]:
            label[2][i] = 1
        if d_label[i][0] in [6,8]:
            label[3][i] = 1
        if d_label[i][0] in [1,3,4,5,7,9]:
            label[4][i] = 1
        if d_label[i][0] in [2,4,6,7]:
            label[5][i] = 1
        if d_label[i][0] in [3,5]:
            label[6][i] = 1
        if d_label[i][0] in [0,1,9]:
            label[7][i] = 1
    
    return label