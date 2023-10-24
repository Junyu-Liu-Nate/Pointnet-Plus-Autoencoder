import os
import numpy as np
import random

from generateWholeFeatures import computeWholeFeature
from fileIO import read_features_from_txt_shapenet, readSelectedMD5s_component, readSelectedMD5s_part, add_pair_to_existing_txt

# BASE_DATA_PATH = "/Users/liujunyu/Data/Research/BVC/ITSS/"
DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

def generatePairsDataset(outputPath):
    dataset = os.path.join(DATASET_PATH, "ComplementMe")
    # datasetTypes = ["components", "parts"]
    datasetTypes = ["parts"]
    category = "Airplane"
    # dataTypes = ["component_all_centered_features", "part_all_centered_features"]
    dataTypes = ["part_all_decompose_features"]
    
    for idx in range(len(datasetTypes)):
        ### Load ShapeNet v2 ComplementMe portion whole shape features (non-stacked!)
        # if idx == 0:
        #     selectedNamePath = os.path.join(DATASET_PATH, "dataset", "ComplementMe", datasetTypes[idx], "Airplane", "component_all_md5s.txt")
        #     selectedWholeNames = readSelectedMD5s_component(selectedNamePath)
        # else:
        #     selectedNamePath = os.path.join(DATASET_PATH, "dataset", "ComplementMe", datasetTypes[idx], "Airplane", "part_all_part_labels.txt")
        #     selectedWholeNames = readSelectedMD5s_part(selectedNamePath)
        selectedNamePath = os.path.join(DATASET_PATH, "ComplementMe", datasetTypes[idx], "Airplane", "part_all_part_labels.txt")
        selectedWholeNames = readSelectedMD5s_part(selectedNamePath)
        
        wholeFeaturesDict = get_all_pointnet_features(selectedWholeNames) # This is non-stacked feature
        print("Finish loading features for ShapeNet whole shapes.")

        featureDataPath = os.path.join(dataset, datasetTypes[idx], category, dataTypes[idx])
        instanceNames = [d for d in os.listdir(featureDataPath) if os.path.isdir(os.path.join(featureDataPath, d))]

        pos_pair_counter = 0
        neg_pair_counter = 0
        for instanceName in instanceNames:
            exact_feature = wholeFeaturesDict[instanceName]

            instanceFolder = os.path.join(featureDataPath, instanceName)
            partNames = [f for f in os.listdir(instanceFolder) if os.path.isfile(os.path.join(instanceFolder, f))]

            for partName in partNames:
                partFeature = np.load(os.path.join(instanceFolder, partName))

                posShapeIdxs = [instanceName]
                num_neighbor = 1
                negShapeIdxs = find_negative_pointnet_neighors(exact_feature, wholeFeaturesDict, num_neighbor)
                print(f"    Positive index: {posShapeIdxs}")
                print(f"    Negative index: {negShapeIdxs}")

                for posShapeIdx in posShapeIdxs:
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
            
            print(f'Finish shape idx: {instanceName}\n')

def find_negative_pointnet_neighors(exact_feature, all_whole_features, num_neighbor):
    distances = {}
    for mesh_path, whole_feature in all_whole_features.items():
        distance = np.linalg.norm(exact_feature - whole_feature)
        distances[mesh_path] = distance
    sorted_mesh_paths = sorted(distances, key=distances.get, reverse=True)

    # Find the midpoint of sorted_mesh_paths
    # midpoint = len(sorted_mesh_paths) // 2
    midpoint = len(sorted_mesh_paths) // 8 #Change to 1/8
    
    # Select only the first half i.e., negative (not similar) part
    negative_half = sorted_mesh_paths[:midpoint]
    
    # Randomly select k mesh_paths from the first half of sorted_mesh_paths
    top_k_mesh_paths = random.sample(negative_half, num_neighbor)

    return top_k_mesh_paths

def get_all_pointnet_features(selectedWholeNames):
    featureDataPath = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/" + "pointnet2_airplane_ShapeNetv2.txt"
    features = read_features_from_txt_shapenet(featureDataPath, selectedWholeNames)
    return features




def main():
    datasetFolder = os.path.join(PROJECT_PATH, )
    outputName = "pairs_ComplementMe_airplane_decompose_exact_oct"
    outputPath = os.path.join(datasetFolder, outputName + ".txt")

    generatePairsDataset(outputPath)

if __name__ == '__main__':
    main()