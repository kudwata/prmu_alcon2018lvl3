import numpy as np
import matplotlib.pyplot as plt

def getScore(N_SAMPLING=32):
    n_sampling = N_SAMPLING

    target = []
    for i in range(8):
        target.append(np.load('model1/model1_{}.npy'.format(i)))

    predicted = np.load('output/output{}.npy'.format(n_sampling))

    sum = 0.0
    acc = 0.0

    for i in range(10000):
        t = 0
        p_f = 0
        r_f = 0
        for j in range(8):
            if predicted[i][j] == 1:
                if np.argmax(target[j][i]) == 1:
                    t+=1
                else:
                    p_f+=1
            else:
                if np.argmax(target[j][i]) == 1:
                    r_f+=1

        if not t == 0:
            p = t/(t+p_f)
            r = t/(t+r_f)
            f = (2*p*r)/(p+r)
            if f == 1:
                acc+=1
            sum+=f

    score = sum/10000
    acc = acc/10000
    
    return score, acc

if __name__ == '__main__':
    samplings = np.array((16,32,64,128,256,512,1024,2048,3072,4096,6144,8192,10000))
    score = np.zeros(samplings.shape[0])
    acc = np.zeros(samplings.shape[0])
    for i in range(samplings.shape[0]):
        score[i], acc[i] = getScore(samplings[i])

    print(score)
    plt.title('AE-kmeans_score')
    plt.plot(samplings, score, label='score')
    plt.plot(samplings, acc, label='accuracy')
    plt.grid(which="both")
    plt.legend()
    plt.show()