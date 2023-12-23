import os
import numpy as np
import random

from generateWholeFeatures_ModelNet40 import computeWholeFeature
from fileIO import read_features_from_txt, create_initial_txt, add_pair_to_existing_txt, read_features_from_txt_shapenet

# BASE_DATA_PATH = "/Users/liujunyu/Data/Research/BVC/ITSS/"
DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

def generatePairsDataset(outputPath):
    datasetFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    shapenetWholeFolder = "ShapeNetv2_Wholes_100"
    meshDatasetFolder = os.path.join(BASE_DATA_PATH, "dataset", "ShapeNetCore_v2", "02691156")
    shapeNames = [d for d in os.listdir(meshDatasetFolder) if os.path.isdir(os.path.join(meshDatasetFolder, d))]

    all_pointnet_features = get_all_pointnet_features()

    create_initial_txt(outputPath)

    pos_pair_counter = 0
    neg_pair_counter = 0
    for shapeName in shapeNames:
        print(f"meshIdx: {shapeName}")

        # exact_feature = all_pointnet_features[shapeIdx]
        exact_feature = all_pointnet_features[shapeName] # Now shapeName is dictionary key

        pointnetWholePath = os.path.join(datasetFolder, shapenetWholeFolder, "airplane", shapeName + ".npy")
        wholeFeature = np.load(pointnetWholePath)

        for partFeature in wholeFeature:
            posShapeIdxs = [shapeName]
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

        print(f'Finish shape idx: {shapeName}\n')


# def find_negative_pointnet_neighors(exact_feature, all_whole_features, num_neighbor):
#     distances = []
#     for whole_feature in all_whole_features:
#         distance = np.linalg.norm(exact_feature - whole_feature)
#         distances.append(distance)
#     enumerated_numbers = list(enumerate(distances))
#     sorted_numbers = sorted(enumerated_numbers, key=lambda x: x[1], reverse = True)

#     # Find the midpoint of sorted_numbers
#     midpoint = len(sorted_numbers) // 2
    
#     # Select only the first half i.e., negative (not similar) part
#     negative_half = sorted_numbers[:midpoint]
#     # Randomly select k indices from the first half of sorted_numbers
#     top_k_indices = random.sample([index for index, _ in negative_half], num_neighbor)

#     return top_k_indices

def find_negative_pointnet_neighors(exact_feature, all_whole_features, num_neighbor):
    distances = {}
    for mesh_path, whole_feature in all_whole_features.items():
        distance = np.linalg.norm(exact_feature - whole_feature)
        distances[mesh_path] = distance
    sorted_mesh_paths = sorted(distances, key=distances.get, reverse=True)

    # Find the midpoint of sorted_mesh_paths
    # midpoint = len(sorted_mesh_paths) // 2
    midpoint = len(sorted_mesh_paths) // 16 #Change to 1/16
    
    # Select only the first half i.e., negative (not similar) part
    negative_half = sorted_mesh_paths[:midpoint]
    
    # Randomly select k mesh_paths from the first half of sorted_mesh_paths
    top_k_mesh_paths = random.sample(negative_half, num_neighbor)

    return top_k_mesh_paths

def get_all_pointnet_features():
    featureDataPath = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/" + "pointnet2_airplane_ShapeNetv2.txt"
    # features = read_features_from_txt(featureDataPath)
    features = read_features_from_txt_shapenet(featureDataPath)
    return features


def main():
    # datasetFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    datasetFolder = os.path.join(PROJECT_PATH, "generated", "ShapeNet")
    outputName = "pairs_airplane_ShapeNet_exact_oct_train"
    outputPath = os.path.join(datasetFolder, outputName + ".txt")

    generatePairsDataset(outputPath)

if __name__ == '__main__':
    main()