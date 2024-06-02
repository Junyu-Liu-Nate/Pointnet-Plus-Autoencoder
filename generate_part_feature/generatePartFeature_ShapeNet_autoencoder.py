import random
import numpy as np
import torch
import os
from tqdm import tqdm
from collections import OrderedDict

from models.pointnet2_cls_ssg import get_model
from train_test_custom.customized_inference import preProcessPC, inferenceBatchAutoencoder, inferenceBatch, preProcessPC_nonormalize
from fileIO import read_obj_vertices, readSelectedMD5s_part

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc

def generatePartFeatures(model, pcDataPath, instanceName, partNames, saveFolder):
    partPCs = []
    for partName in partNames:
        partPath = os.path.join(pcDataPath, instanceName, partName)
        point_set = read_obj_vertices(partPath)
        if len(point_set) == 0:
            continue # Remove empty part
        point_set = np.array(point_set)
        
        # partProcessed = preProcessPC(2048, partVertices)
        # partProcessed = preProcessPC_nonormalize(2048, partVertices)
        num_points = 2048
        if len(point_set) < num_points:
            print(f"pad {len(point_set)}")
            # Pad the point_set
            padding = np.zeros((num_points - len(point_set), point_set.shape[1]))
            point_set = np.vstack((point_set, padding))
        elif len(point_set) > num_points:
            print(f"random sample {len(point_set)}")
            # Randomly sample or trim
            choice = np.random.choice(len(point_set), num_points, replace=False)
            point_set = point_set[choice, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        partPCs.append(point_set)
    if len(partPCs) == 0:
        return
    partPCs = np.array(partPCs)

    featureArray = inferenceBatchAutoencoder(model, partPCs)

    for partIdx in range(len(featureArray)):
        saveInstancePath = os.path.join(saveFolder, instanceName)
        if not os.path.exists(saveInstancePath):
            os.makedirs(saveInstancePath)
        savePath = os.path.join(saveInstancePath, str(partIdx) + ".npy") ### Index!!!
        np.save(savePath, featureArray[partIdx])


def main():
    #%% Load PointNet model
    modelPath = 'log/reconstruction/Spaghetti_decomposition_grouped/checkpoints/best_model.pth'
    print("Model loaded.")
    # Initialize the model and load the trained weights
    model = get_model(2048, -1, False)
    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    # model.load_state_dict(model_state_dict)

    # Adjusting keys - For model trained with data parallel
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
        new_state_dict[name] = v
    # Loading the adjusted state dict
    model.load_state_dict(new_state_dict)
    
    model.eval()

    #%% ShapeNet/Spaghetti Dataset
    dataset = os.path.join(PROJECT_PATH, "generated", "Spaghetti")
    datasetType = "spaghetti_decomposition_grouped_fps2048" # ShapeNet_ACD_16
    category = "02691156"
    # pcDataPath = os.path.join(dataset, datasetType, category)
    pcDataPath = os.path.join(dataset, datasetType)

    #%%
    saveFolderName = "spaghetti_decomposition_grouped_features"
    # saveFolder = os.path.join(dataset, saveFolderName, "02691156")
    saveFolder = os.path.join(dataset, saveFolderName)

    #%% Generate and save PointNet features
    # instanceNames = [d for d in os.listdir(pcDataPath) if os.path.isdir(os.path.join(pcDataPath, d))]
    selectedNamePath = os.path.join(PROJECT_PATH, "generated", "Spaghetti", 'plane_names_all.txt')
    instanceNames = readSelectedMD5s_part(selectedNamePath)
    instanceNames = tqdm(instanceNames)
    for instanceName in instanceNames:
        saveInstancePath = os.path.join(saveFolder, instanceName)
        if not os.path.exists(saveInstancePath):
            instanceNames.set_description(f"Start generating features for shape {instanceName}")
            instanceFolder = os.path.join(pcDataPath, instanceName)
            if not os.path.exists(instanceFolder):
                continue
            partNames = [f for f in os.listdir(instanceFolder) if os.path.isfile(os.path.join(instanceFolder, f)) and not f.startswith("._")]

            generatePartFeatures(model, pcDataPath, instanceName, partNames, saveFolder)
        else:
            instanceNames.set_description(f"Skip generating features for shape {instanceName}")

if __name__ == '__main__':
    main()