import os
import numpy as np

from generateWholeFeatures import computeWholeFeature
from fileIO import read_features_from_txt

def generatePairsDataset(outputPath, wholeMeshPaths):
    datasetFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    pointnetWholeFolder = "Pointnet_Wholes"
    pointnetWholeObject = "airplane"

    all_pointnet_features = get_all_pointnet_features()

    for shapeIdx in range(len(wholeMeshPaths)):
        meshIdx = int(wholeMeshPaths[shapeIdx][-8:-4])
        print(f"meshIdx: {meshIdx}")

        exact_feature = all_pointnet_features[shapeIdx]

        pointnetWholePath = os.path.join(datasetFolder, pointnetWholeFolder, pointnetWholeObject, str(meshIdx) + ".npy")
        wholeFeature = np.load(pointnetWholePath)

        for partFeature in wholeFeature:
            positive_whole_path_idxs = [shapeIdx]
            negative_whole_path_idxs = find_negative_pointnet_neighors(exact_feature, all_pointnet_features, num_neighbor)

def get_all_pointnet_features():
    featureDataPath = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/" + "pointnet2_airplane_600.txt"
    features = read_features_from_txt(featureDataPath)
    return features

def main():
    pass

if __name__ == '__main__':
    main()