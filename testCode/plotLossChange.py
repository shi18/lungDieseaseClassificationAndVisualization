import matplotlib
import matplotlib.pyplot as plt
import numpy as np

withoutAugPath = '/Users/shiyan/Desktop/recod/dataset8/'
augPath = '/Users/shiyan/Desktop/recod/dataset8Aug/'


def getAvg(path):
    npy1 = '01epoch64lr00001batch128_epochLoss.npy'
    npy2 = '02epoch64lr00001batch128_epochLoss.npy'
    npy3 = '03epoch64lr00001batch128_epochLoss.npy'
    npy4 = '04epoch64lr00001batch128_epochLoss.npy'
    npy5 = '05epoch64lr00001batch128_epochLoss.npy'
    npy6 = '06epoch64lr00001batch128_epochLoss.npy'
    npy7 = '07epoch64lr00001batch128_epochLoss.npy'
    npy8 = '08epoch64lr00001batch128_epochLoss.npy'
    npy9 = '09epoch64lr00001batch128_epochLoss.npy'
    npy10 = '10epoch64lr00001batch128_epochLoss.npy'
    y1 = np.load(path + npy1)
    y2 = np.load(path + npy2)
    y3 = np.load(path + npy3)
    y4 = np.load(path + npy4)
    y5 = np.load(path + npy5)
    y6 = np.load(path + npy6)
    y7 = np.load(path + npy7)
    y8 = np.load(path + npy8)
    y9 = np.load(path + npy9)
    y10 = np.load(path + npy10)

    zeroList = np.zeros(64).tolist()

    for i in range(64):
        zeroList[i] = (y1[i] + y2[i] + y3[i] + y4[i] + y5[i] + y6[i] + y7[i] + y8[i] + y9[i] + y10[i]) / 10.0
    return zeroList


# withoutAugAvgLoss = getAvg(withoutAugPath)
augAvgLoss = getAvg(augPath)

x = np.arange(start=0, stop=64, step=1, dtype='uint').tolist()
np0 = './epoch64lr0001batch128_epochLoss.npy'
y0 = np.load(np0)
npc = './epoch64lr000001batch128_epochLoss.npy'
yc = np.load(npc)

l0 = plt.plot(x, y0, 'y+-', label='lr = 0.0001')
# l1 = plt.plot(x, withoutAugAvgLoss, 'b--', label='lr = 0.00001 on original data')
l2 = plt.plot(x, augAvgLoss, 'r--', label='lr = 0.00001')
l3 = plt.plot(x, yc, 'g+-', label='lr = 0.000001')
# plt.plot(x,y1,'ro-',x,y2,'g+-',x,y3,'b^-')
# plt.plot(x, y0, 'y^-', x, withoutAugAvgLoss, 'b--', x, augAvgLoss, 'r--', x, yc, 'g+-')
plt.plot(x, y0, 'y+-', x, augAvgLoss, 'r--', x, yc, 'g+-')

plt.title('The average loss change with 64 epochs 128 batchsize')
plt.xlabel('epoch')
plt.ylabel('aveLoss')
plt.legend()
plt.savefig('lossRecord.png')
plt.show()
