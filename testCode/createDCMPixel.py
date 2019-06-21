import copy
import numpy as np
import SimpleITK as sitk
import sys
import matplotlib.pyplot as plt
from math import floor, ceil
import collections

def get3DArry(path):
    seriesIDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
    if not seriesIDs:
        print("ERROR: given directory \"" + path + "\" does not contain a DICOM series.")
        sys.exit(1)
    seriesFileNames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, seriesIDs[0])
    # print(seriesFileNames)

    seriesReader = sitk.ImageSeriesReader()
    seriesReader.SetFileNames(seriesFileNames)
    img = seriesReader.Execute()
    # shape of img3D is (layers,512,512)
    img3D = sitk.GetArrayViewFromImage(img)
    return img3D


def saveITK(image, filename):
    im = sitk.GetImageFromArray(image, isVector=False)
    sitk.WriteImage(im, filename, True)

def getMaxNeigh(maskCopy, layer, r, c, step):
    center = maskCopy[layer, :, :][r, c]
    neigh1 = maskCopy[layer, :, :][r - step, c]
    neigh2 = maskCopy[layer, :, :][r, c - step]
    neigh3 = maskCopy[layer, :, :][r, c + step]
    neigh4 = maskCopy[layer, :, :][r + step, c]

    neigh5 = maskCopy[layer, :, :][r, c - 2 * step]
    neigh6 = maskCopy[layer, :, :][r, c + 2 * step]
    neigh7 = maskCopy[layer, :, :][r - 2 * step, c]
    neigh8 = maskCopy[layer, :, :][r + 2 * step, c]

    neigh9 = maskCopy[layer, :, :][r - step, c - step]
    neigh10 = maskCopy[layer, :, :][r - step, c + step]
    neigh11 = maskCopy[layer, :, :][r + step, c - step]
    neigh12 = maskCopy[layer, :, :][r + step, c + step]
    neighArr = np.asarray([neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7, neigh8, neigh9, neigh10, neigh11,
                           neigh12])
    # print('neighArr:', neighArr)
    countRes = collections.Counter(neighArr).most_common(2)
    # print(countRes)
    first = countRes[0][0]
    # print(first, sec)
    if first != 0:
        maxVal = first
    elif first == 0 and len(countRes) > 1:
        sec = countRes[1][0]
        maxVal = sec
    else:
        maxVal = center

    return maxVal, maskCopy

def denoise(lenOfPatientLUCList, patientLUCList, maskCopy3, maskCopy, step):
    for layer in range(lenOfPatientLUCList):
        tmp = patientLUCList[layer]
        tmpLength = len(tmp)
        if tmpLength != 0:
            for j in range(tmpLength):
                r = tmp[j][0]
                c = tmp[j][1]
                tmpMask2 = maskCopy3[layer, :, :][r - floor(step / 2): r + ceil(step / 2),
                           c - floor(step / 2): c + ceil(step / 2)]
                maxVal, maskCopy = getMaxNeigh(maskCopy, layer, r, c, step)
                maskCopy[layer, :, :][r - floor(step / 2): r + ceil(step / 2),
                c - floor(step / 2): c + ceil(step / 2)] = maxVal
    return maskCopy
def generate(patientID, step):
    oriPath = '/Users/shiyan/Downloads/lung/data/testData/' + str(patientID) + '/'
    maskPath = oriPath + 'lung_mask/'
    maskImg = get3DArry(maskPath)
    maskCopy = copy.deepcopy(maskImg)
    maskCopy2 = copy.deepcopy(maskImg)
    maskCopy3 = copy.deepcopy(maskImg)
    maskCopy[maskCopy != 0] = 0

    patientLUCList = np.load('./patient' + str(patientID) + 'LUCList.npy')
    lenOfPatientLUCList = len(patientLUCList)
    predicted = np.load('./predicted36.npy')

    count = 0
    for layer in range(lenOfPatientLUCList):
        # print('layer', layer)
        tmp = patientLUCList[layer]
        tmpLength = len(tmp)

        if tmpLength != 0:
            # print('layer:', layer + 1)
            for j in range(tmpLength):
                label = predicted[count]
                count += 1
                r = tmp[j][0]
                c = tmp[j][1]
                tmpMask = maskCopy2[layer, :, :][r - floor(step / 2): r + ceil(step / 2),
                          c - floor(step / 2): c + ceil(step / 2)]
                # tmpMask[tmpMask == 255] = label + 1
                # print(r, c)
                maskCopy[layer, :, :][r - floor(step / 2): r + ceil(step / 2),
                c - floor(step / 2): c + ceil(step / 2)] = label + 1
    maskCopy = denoise(lenOfPatientLUCList, patientLUCList, maskCopy3, maskCopy, step)
    maskCopy = denoise(lenOfPatientLUCList, patientLUCList, maskCopy3, maskCopy, step)
    maskCopy = denoise(lenOfPatientLUCList, patientLUCList, maskCopy3, maskCopy, step)
    maskCopy = denoise(lenOfPatientLUCList, patientLUCList, maskCopy3, maskCopy, step)

    # for layer in range(lenOfPatientLUCList):
    #     tmp = patientLUCList[layer]
    #     tmpLength = len(tmp)
    #     if tmpLength != 0:
    #         for j in range(tmpLength):
    #             r = tmp[j][0]
    #             c = tmp[j][1]
    #             tmpMask2 = maskCopy3[layer, :, :][r - floor(step / 2): r + ceil(step / 2),
    #                        c - floor(step / 2): c + ceil(step / 2)]
    #             maxVal, maskCopy = getMaxNeigh(maskCopy, layer, r, c, step)
    #             maskCopy[layer, :, :][r - floor(step / 2): r + ceil(step / 2),
    #             c - floor(step / 2): c + ceil(step / 2)] = maxVal



    print(count)
    split = oriPath.split('/')
    id = split[-2]
    dcmName = id + 'mask.nii'

    saveITK(maskCopy, dcmName)


generate(patientID=36, step=5)
