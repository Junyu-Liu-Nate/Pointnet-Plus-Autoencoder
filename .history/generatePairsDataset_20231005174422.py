import os
import numpy as np
import random

from generateWholeFeatures import computeWholeFeature
from fileIO import read_features_from_txt, create_initial_txt, add_pair_to_existing_txt

### Note:
### Currently the parts are directly extracted from the whole features (each whole contains n part features)
###  -  "Pointnet_Wholes/" and "Pointnet_Wholes_test/" whole features contain 50 parts
###     - 
def generatePairsDataset(outputPath):
    datasetFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    pointnetWholeFolder = "Pointnet_Wholes_test"
    pointnetWholeObject = "airplane"

    all_pointnet_features = get_all_pointnet_features()

    create_initial_txt(outputPath)

    pos_pair_counter = 0
    neg_pair_counter = 0
    for shapeIdx in range(600):
        meshIdx = shapeIdx
        print(f"meshIdx: {meshIdx}")

        exact_feature = all_pointnet_features[shapeIdx]

        pointnetWholePath = os.path.join(datasetFolder, pointnetWholeFolder, pointnetWholeObject, str(meshIdx) + ".npy")
        wholeFeature = np.load(pointnetWholePath)

        for partFeature in wholeFeature:
            posShapeIdxs = [shapeIdx]
            num_neighbor = 1
            negShapeIdxs = find_negative_pointnet_neighors(exact_feature, all_pointnet_features, num_neighbor)
            print(f"    Positive index: {posShapeIdxs} (remember to +1 when finding files)")
            print(f"    Negative index: {negShapeIdxs} (remember to +1 when finding files)")

            for posShapeIdx in posShapeIdxs:
                #TODO: Need to modify the read fucntion when loading in training
                pair = {
                    'id': str(pos_pair_counter),
                    'label': True,
                    'part': partFeature,
                    'whole': str(posShapeIdx)
                }
                add_pair_to_existing_txt(outputPath, pair)
                pos_pair_counter += 1
            for negShapeIdx in negShapeIdxs:
                pair = {
                    'id': str(neg_pair_counter),
                    'label': False,
                    'part': partFeature,
                    'whole': str(negShapeIdx)
                }
                add_pair_to_existing_txt(outputPath, pair)
                neg_pair_counter += 1

        print(f'Finish shape idx: {shapeIdx}\n')


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
    # Randomly select k indices from the first half of sorted_numbers
    top_k_indices = random.sample([index for index, _ in negative_half], num_neighbor)

    return top_k_indices

def get_all_pointnet_features():
    featureDataPath = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/" + "pointnet2_airplane_600.txt"
    features = read_features_from_txt(featureDataPath)
    return features

def main():
    datasetFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    outputName = "pairs_ModelNet40_airplane_PointNet_600_exact_test"
    outputPath = os.path.join(datasetFolder, outputName + ".txt")

    generatePairsDataset(outputPath)

if __name__ == '__main__':
    main()