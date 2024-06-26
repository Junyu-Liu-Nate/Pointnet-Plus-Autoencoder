import random
import numpy as np
import torch
import os
from tqdm import tqdm

from models.pointnet2_cls_ssg_original import get_model
from train_test_custom.customized_inference import preProcessPC, inferenceBatch, preProcessPC_nonormalize
from fileIO import read_obj_vertices, get_names_from_json

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

def generatePartFeatures(model, pcDataPath, instanceName, partNames, saveFolder):
    partPCs = []
    for partName in partNames:
        partPath = os.path.join(pcDataPath, instanceName, partName)
        partVertices = read_obj_vertices(partPath)
        if len(partVertices) == 0:
            continue # Remove empty part
        partVertices = np.array(partVertices)
        partProcessed = preProcessPC(1024, partVertices)
        # partProcessed = preProcessPC_nonormalize(partVertices)
        partPCs.append(partProcessed)
    if len(partPCs) == 0:
        return
    partPCs = np.array(partPCs)

    featureArray = inferenceBatch(model, partPCs)

    for partIdx in range(len(featureArray)):
        saveInstancePath = os.path.join(saveFolder, instanceName)
        if not os.path.exists(saveInstancePath):
            os.makedirs(saveInstancePath)
        savePath = os.path.join(saveInstancePath, str(partIdx + 1) + ".npy")
        np.save(savePath, featureArray[partIdx])


def main():
    #%% Load PointNet model
    modelPath = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    print("Model loaded.")
    # Initialize the model and load the trained weights
    model = get_model(40, False)
    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    #%% ShapeNet ACD / PartNet Dataset
    dataset = os.path.join(PROJECT_PATH, "generated", "ShapeNet")
    datasetType = "ShapeNetv2_Wholes_sphere_50_normalized_PC" # ShapeNet_ACD_16
    category = "02691156"
    pcDataPath = os.path.join(dataset, datasetType, category)

    #%%
    saveFolderName = "ShapeNet_sphere_50_features" # ShapeNet_ACD_16_features
    saveFolder = os.path.join(dataset, saveFolderName, "02691156")

    #%% Generate and save PointNet features
    instanceNames = [d for d in os.listdir(pcDataPath) if os.path.isdir(os.path.join(pcDataPath, d))]
    # spaghettiNamePath = os.path.join(PROJECT_PATH, "generated", "Spaghetti", "shapenet_chairs_train.json")
    # instanceNames = get_names_from_json(spaghettiNamePath, "03001627")
    instanceNames = tqdm(instanceNames)
    for instanceName in instanceNames:
        saveInstancePath = os.path.join(saveFolder, instanceName)
        if not os.path.exists(saveInstancePath):
            instanceNames.set_description(f"Start generating features for shape {instanceName}")
            instanceFolder = os.path.join(pcDataPath, instanceName)
            partNames = [f for f in os.listdir(instanceFolder) if os.path.isfile(os.path.join(instanceFolder, f)) and not f.startswith("._")]

            generatePartFeatures(model, pcDataPath, instanceName, partNames, saveFolder)
        else:
            instanceNames.set_description(f"Skip generating features for shape {instanceName}")

if __name__ == '__main__':
    main()