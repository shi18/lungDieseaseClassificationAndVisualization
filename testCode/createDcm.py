
import copy
import numpy as np
import SimpleITK as sitk
import sys
import matplotlib.pyplot as plt

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

def generate (patientID):

    oriPath = '/Users/shiyan/Downloads/lung/data/testData/'+str(patientID)+'/'
    maskPath = oriPath + 'lung_mask/'
    maskImg = get3DArry(maskPath)
    maskCopy = copy.deepcopy(maskImg)
    maskCopy[maskCopy != 0] = 0

    patientLUCList = np.load('./patient'+str(patientID)+'LUCList.npy')
    lenOfPatientLUCList = len(patientLUCList)
    predicted = np.load('./predicted.npy')


    count = 0
    for layer in range(lenOfPatientLUCList):
        tmp = patientLUCList[layer]
        tmpLength = len(tmp)

        if tmpLength != 0:
            for j in range(tmpLength):
                label = predicted[count]
                count += 1

                r = tmp[j][0]
                c = tmp[j][1]
                maskCopy[layer, :, :][r: r + 32, c: c + 32] = label+1
    split = oriPath.split('/')
    id = split[-2]
    dcmName = id+'mask.nii'

    saveITK(maskCopy, dcmName)

generate(39)