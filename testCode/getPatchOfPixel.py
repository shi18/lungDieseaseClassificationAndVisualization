import os, sys, glob
import SimpleITK as sitk
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt


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


def getSatifiedPatchFlag(maskSlice, threshold, idx, step):
    leftUpCoorList = []
    # select patches which are satisfied our requirement
    leftMost = [(0, 0)]
    for idxr in range(512):
        for idxc in range(512):
            if maskSlice[idxr, idxc] == 255:
                if leftMost[-1][0] != idxr:
                    leftMost.append((idxr, idxc))
                continue
        continue
    del leftMost[0]
    for row in range(0, len(leftMost), step):
            for c in range(leftMost[row][1], 512, step):
                r = leftMost[row][0]
                if maskSlice[r, c] != 0:
                    oriRow = r - 15
                    oriCol = c - 15
                    patch = maskSlice[oriRow:oriRow + 32, oriCol:oriCol + 32]
                    # decide whether the patch is satified by a threshold
                    arrSum = np.sum(patch)
                    # if whole patch in mask, then sum of patch is 32*32 =1024, 768 is 75% percent.
                    if arrSum >= threshold * 255:
                        a, b = patch.shape
                        leftUpCoorList.append((r, c))
    if len(leftUpCoorList) != 0:
        print(idx, len(leftUpCoorList))
        # plt.imshow(maskSlice, cmap='gray')
        # plt.show()
    return leftUpCoorList


def extractPatchFromOri(leftUpCoorList, oriSlice, count, saveFolderPath):
    oriPatchList = []
    for i in range(len(leftUpCoorList)):
        r = leftUpCoorList[i][0]
        c = leftUpCoorList[i][1]
        oriPatch = oriSlice[r-15:r + 17, c-15:c + 17]
        oriPatchList.append(oriPatch)
    # convert intensity image into rgb image by using spercified rules
    # and save image as .png files.
    # print('oriPatchList: ', len(oriPatchList))
    if count < 10:
        sliceID = 'Slice00' + str(count)
    else:
        sliceID = 'Slice0' + str(count)

    for i in range(len(oriPatchList)):
        tmp = oriPatchList[i]
        out = threeWindows(oriPatchList[i])
        adjusted = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        # set image name
        if i < 9:
            patchID = 'Patch00000' + str(i + 1)
        elif i < 99:
            patchID = 'Patch0000' + str(i + 1)
        elif i < 999:
            patchID = 'Patch000' + str(i + 1)
        elif i < 9999:
            patchID = 'Patch00' + str(i + 1)
        elif i < 99999:
            patchID = 'Patch0' + str(i + 1)
        imgName = saveFolderPath + sliceID + patchID + '.png'
        # print(imgName)
        cv2.imwrite(imgName, adjusted)


def saveITK(image, filename):
    im = sitk.GetImageFromArray(image, isVector=False)
    sitk.WriteImage(im, filename, True)


def processOnePatient(patientID, threshold, step):
    oriPath = '**/lung/data/testData/' + str(patientID) + '/'
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

        leftUpCoorList = getSatifiedPatchFlag(maskSlice, threshold, idx, step)
        patientLUCList.append(leftUpCoorList)
        flag = len(leftUpCoorList)
        if flag != 0:
            extractPatchFromOri(leftUpCoorList, oriSlice, count, saveFolderPath='../1/')
    return patientLUCList


import glob, os, os.path


def saveList(patientID, threshold, step):
    filelist = glob.glob(os.path.join('../1', "*.png"))
    for f in filelist:
        os.remove(f)
    patientLUCList = processOnePatient(patientID, threshold, step)
    sum = 0
    for lenIdx in range(len(patientLUCList)):
        a = len(patientLUCList[lenIdx])
        sum += a
    print(sum)
    npyName = 'patient' + str(patientID) + 'LUCList.npy'
    np.save(npyName, patientLUCList)

# 17, 23, 36, 38, 39, 41, 48, 53, 57, 70
saveList(patientID=36, threshold=896, step=5)
