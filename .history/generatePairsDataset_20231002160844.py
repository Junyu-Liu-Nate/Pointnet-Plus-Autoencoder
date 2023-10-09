import os
import numpy as np

from generateWholeFeatures import computeWholeFeature

def generatePairsDataset(outputPath, wholeMeshPaths):
    datasetFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    pointnetWholeFolder = "Pointnet_Wholes"
    pointnetWholeObject = "airplane"

    for wholeMeshPath in wholeMeshPaths:
        meshIdx = int(wholeMeshPath[-8:-4])
        print(f"meshIdx: {meshIdx}")

        pointnetWholePath = os.path.join(datasetFolder, pointnetWholeFolder, pointnetWholeObject, str(meshIdx) + ".npy")
        wholeFeature = np.load(pointnetWholePath)

        for partFeature in wholeFeature


def main():
    pass

if __name__ == '__main__':
    main()