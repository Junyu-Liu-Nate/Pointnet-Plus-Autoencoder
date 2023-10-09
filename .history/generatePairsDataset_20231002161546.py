import os
import numpy as np
import random

from generateWholeFeatures import computeWholeFeature
from fileIO import read_features_from_txt

def generatePairsDataset(outputPath, wholeMeshPaths):
    datasetFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    pointnetWholeFolder = "Pointnet_Wholes"
    pointnetWholeObject = "airplane"

    all_pointnet_features = get_all_pointnet_features()

    for shapeIdx in range(len(wholeMeshPaths)):
        meshIdx = shapeIdx + 1
        print(f"meshIdx: {meshIdx}")

        exact_feature = all_pointnet_features[shapeIdx]

        pointnetWholePath = os.path.join(datasetFolder, pointnetWholeFolder, pointnetWholeObject, str(meshIdx) + ".npy")
        wholeFeature = np.load(pointnetWholePath)

        for partFeature in wholeFeature:
            positive_whole_path_idxs = [shapeIdx]
            num_neighbor = 1
            negative_whole_path_idxs = find_negative_pointnet_neighors(exact_feature, all_pointnet_features, num_neighbor)

def find_negative_pointnet_neighors(exact_feature, all_whole_features, num_neighbor):
    distances = []
    for whole_feature in all_whole_features:
        distance = np.linalg.norm(exact_feature - whole_feature)
        distances.append(distance)
    enumerated_numbers = list(enumerate(distances))
    sorted_numbers = sorted(enumerated_numbers, key=lambda x: x[1], reverse = True)

    # Find the midpoint of sorted_numbers
    midpoint = len(sorted_numbers) // 2
    
    # Select only the first half i.e., negative (not similar) part
    negative_half = sorted_numbers[:midpoint]
    # print(negative_half)
    # Randomly select k indices from the first half of sorted_numbers
    top_k_indices = random.sample([index for index, _ in negative_half], num_neighbor)

    # top_k_indices = [index for index, _ in sorted_numbers[:num_neighbor]]
    return top_k_indices

def get_all_pointnet_features():
    featureDataPath = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/" + "pointnet2_airplane_600.txt"
    features = read_features_from_txt(featureDataPath)
    return features

def main():
    pass

if __name__ == '__main__':
    main()