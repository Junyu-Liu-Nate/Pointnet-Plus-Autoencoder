import os
import numpy as np
import random

from generateWholeFeatures import computeWholeFeature
from fileIO import readWholePointnetFeatures_ShapeNet, readSelectedMD5s, add_pair_to_existing_txt

BASE_DATA_PATH = "/Users/liujunyu/Data/Research/BVC/ITSS/"

def generatePairsDataset(outputPath):
    #%% Load pre-generated features for whole shapes
    ### Load ShapeNet v2 ComplementMe portion whole shape features
    whole_feature_path = os.path.join(BASE_DATA_PATH, "dataset", "ShapeNetv2_Wholes_100", "airplane")
    selectedNamePath = os.path.join(BASE_DATA_PATH, "dataset", "ComplementMe", "components", "Airplane", "component_all_md5s.txt")
    selectedWholeNames = readSelectedMD5s(selectedNamePath)

    wholeFeaturesDict = readWholePointnetFeatures_ShapeNet(whole_feature_path, selectedWholeNames)
    wholeNames = list(wholeFeaturesDict.keys())
    wholeFeatures = list(wholeFeaturesDict.values())
    print("Finish loading features for ShapeNet whole shapes.")

    #%% ComplementMe Dataset
    dataset = os.path.join(BASE_DATA_PATH, "dataset", "ComplementMe")
    datasetType = "parts"
    category = "Airplane"
    dataType = "part_all_centered_point_clouds"
    featureDataPath = os.path.join(dataset, datasetType, category, dataType)

    instanceNames = [d for d in os.listdir(featureDataPath) if os.path.isdir(os.path.join(featureDataPath, d))]

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