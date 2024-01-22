import numpy as np
import torch
import os
from tqdm import tqdm

from models.pointnet2_cls_ssg import get_model
from customized_inference import preProcessPC, inferenceBatchAutoencoder, inferenceBatchReconstructAutoencoder
from fileIO import read_obj_vertices, savePCAsObj

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

def reconstruction(model, pcDataPath, instanceName, partNames, saveFolder):
    partPCs = []
    for partName in partNames:
        partPath = os.path.join(pcDataPath, instanceName, partName)
        partVertices = read_obj_vertices(partPath)
        if len(partVertices) == 0:
            continue # Remove empty part
        partVertices = np.array(partVertices)
        partProcessed = preProcessPC(2048, partVertices)
        partPCs.append(partProcessed)
    if len(partPCs) == 0:
        return
    partPCs = np.array(partPCs)

    reconstructPCs = inferenceBatchReconstructAutoencoder(model, partPCs)

    for partIdx in range(len(reconstructPCs)):
        saveInstancePath = os.path.join(saveFolder, instanceName)
        if not os.path.exists(saveInstancePath):
            os.makedirs(saveInstancePath)
        savePath = os.path.join(saveInstancePath, str(partIdx + 1) + ".obj")
        # np.save(savePath, featureArray[partIdx])
        savePCAsObj(reconstructPCs[partIdx], savePath)



def main():
    #%% Load PointNet model
    modelPath = 'log/reconstruction/shapeNet_FPS_parts_no_aug/checkpoints/best_model.pth'
    print("Model loaded.")
    # Initialize the model and load the trained weights
    model = get_model(2048, -1, False)
    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    #%% ComplementMe Dataset
    # dataset = os.path.join(DATASET_PATH, "ComplementMe")
    # datasetType = "parts"
    # category = "Airplane"
    # dataType = "part_all_original_decompose_resample_point_clouds_v3"
    # pcDataPath = os.path.join(dataset, datasetType, category, dataType)

    # saveFolderName = "part_all_decompose_resample_reconstruct_v3_autoencoder"
    # saveFolder = os.path.join(dataset, datasetType, category, saveFolderName)

    #%% ShapeNet FPS cut 100 dataset (used for ShapeNet wholes)
    pcDataPath = os.path.join(PROJECT_PATH, "generated", "ShapeNet", "ShapeNetv2_Wholes_ellipsoid_100_PC", "02691156")

    saveFolderName = "ShapeNetv2_Wholes_ellipsoid_100_PC_reconstruct_autoencoder"
    saveFolder = os.path.join(PROJECT_PATH, "generated", "ShapeNet", saveFolderName)

    #%% Generate and save PointNet features
    instanceNames = [d for d in os.listdir(pcDataPath) if os.path.isdir(os.path.join(pcDataPath, d))]
    instanceNames = tqdm(instanceNames)
    for instanceName in instanceNames:
        instanceNames.set_description(f"Start generating features for shape {instanceName}")
        instanceFolder = os.path.join(pcDataPath, instanceName)
        partNames = [f for f in os.listdir(instanceFolder) if os.path.isfile(os.path.join(instanceFolder, f)) and not f.startswith("._")]

        reconstruction(model, pcDataPath, instanceName, partNames, saveFolder)


if __name__ == '__main__':
    main()