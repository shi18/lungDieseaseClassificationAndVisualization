import matplotlib

# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os, sys, glob
import SimpleITK as sitk
import numpy as np
import cv2
import copy


def rescale(inArr, a, b):
    m = inArr.min()
    M = inArr.max()
    numerator = (b - a) * (inArr - m)
    denominator = (M - m)
    out = (numerator / denominator) + a
    return out

def threeWindows(inArr):
    tmpArr = np.zeros(32 * 32 * 3).reshape(32, 32, 3)
    if (inArr.min() == 0):
        inArr = inArr - 1000
    elif (inArr.min() == -2000):
        inArr[inArr == -2000] = 0
        inArr = inArr - 1024
    ctWin = np.array([[-160, 240], [-1000, 200], [-1000, -550]])

    for w in range(3):
        winMin = ctWin[w, 0]
        winMax = ctWin[w, 1]
        I = copy.deepcopy(inArr)
        I[I <= winMin] = winMin
        I[I >= winMax] = winMax
        temp = np.round(rescale(I, 0, 255))
        tmpArr[:, :, w] = rescale(I, 0, 255)
    outArr = np.round(tmpArr).astype('uint8')

    return outArr


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


def getSatifiedPatchFlag(maskSlice, threshold):
    flagList = []
    # select patches which are satisfied our requirement
    for r in range(0, 512, 32):
        for c in range(0, 512, 32):
            patch = maskSlice[r:r + 32, c:c + 32]
            # decide whether the patch is satified by a threshold
            arrSum = np.sum(patch)
            # if whole patch in mask, then sum of patch is 32*32 =1024, 768 is 75% percent.
            if arrSum >= threshold*255:
                flagList.append(1)
            else:
                flagList.append(0)
    return flagList


def getSatifiedPatchUpperLeftCoord(flagList):
    flagArr = np.array(flagList)
    idx = np.where(flagArr == 1)[0]
    idxLen = idx.shape[0]
    leftUpCoorList = []
    # convert index into coordinate of the upper-left corner of the patch
    if idxLen !=0 :
        for i in range(idxLen):
            quotient = int(idx[i] / 16)
            remainder = idx[i] % 16
            leftUp = [quotient * 32, remainder * 32]
            leftUpCoorList.append(leftUp)
        return leftUpCoorList
    else:
        return leftUpCoorList


def extractPatchFromOri(leftUpCoorList, oriSlice, count, saveFolderPath):
    oriPatchList = []
    for i in range(len(leftUpCoorList)):
        r = leftUpCoorList[i][0]
        c = leftUpCoorList[i][1]
        oriPatch = oriSlice[r:r + 32, c:c + 32]
        oriPatchList.append(oriPatch)
    # convert intensity image into rgb image by using spercified rules
    # and save image as .png files.
    # print('oriPatchList: ', len(oriPatchList))
    if count < 10:
        sliceID = 'Slice0' + str(count)
    else:
        sliceID = 'Slice' + str(count)
    for i in range(len(oriPatchList)):
        tmp = oriPatchList[i]
        out = threeWindows(oriPatchList[i])
        adjusted = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        # set image name
        if i < 9:
            patchID = 'Patch0' + str(i + 1)
        else:
            patchID = 'Patch' + str(i + 1)
        imgName = saveFolderPath + sliceID + patchID + '.png'
        # print(imgName)
        cv2.imwrite(imgName, adjusted)

def saveITK(image, filename):
    im = sitk.GetImageFromArray(image, isVector=False)
    sitk.WriteImage(im, filename, True)

def processOnePatient(patientID):
    oriPath = '**/lung/data/testData/'+str(patientID)+'/'
    maskPath = oriPath + 'lung_mask/'
    oriImg = get3DArry(oriPath).astype('int64')
    maskImg = get3DArry(maskPath)
    sliceNum, _, _ = maskImg.shape
    # print('sliceNum:', sliceNum)
    count = 0
    patientLUCList = []
    for idx in range(sliceNum):
        count += 1
        oriSlice = oriImg[idx, :, :]
        oriSlice = oriSlice.astype('int64')
        maskSlice = maskImg[idx, :, :]
        # plt.imshow(oriSlice, cmap='gray')
        # plt.imshow(maskSlice, cmap='gray')
        # plt.show()
        # tmpMaskSlice = copy.deepcopy(maskSlice)
        # tmpMaskSlice[tmpMaskSlice > 0] = 1

        flagList = getSatifiedPatchFlag(maskSlice, 896)
        leftUpCoorList = getSatifiedPatchUpperLeftCoord(flagList)
        patientLUCList.append(leftUpCoorList)
        flag = len(leftUpCoorList)
        if flag != 0:
            extractPatchFromOri(leftUpCoorList, oriSlice, count, saveFolderPath='./testPatches/')
    return patientLUCList

import glob, os, os.path

def saveList(patientID):
    filelist = glob.glob(os.path.join('./testPatches', "*.png"))
    for f in filelist:
        os.remove(f)
    patientLUCList = processOnePatient(patientID)
    npyName = 'patient' + str(patientID) + 'LUCList.npy'
    np.save(npyName, patientLUCList)
saveList(39)
