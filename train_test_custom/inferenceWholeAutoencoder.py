import numpy as np
import torch
import os
from tqdm import tqdm

from models.pointnet2_cls_ssg import get_model
from customized_inference import preProcessPC, inferenceBatchAutoencoder, inferenceBatchReconstructAutoencoder
from fileIO import read_obj_vertices, savePCAsObj, readSelectedMD5s_part

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

def reconstruction(model, instanceName, pcDataPath, saveFolder):    
    pcPath = os.path.join(pcDataPath, instanceName)
    pc = read_obj_vertices(pcPath)
    pcProcessed = preProcessPC(2048, pc)
    pcProcessed = np.array([pcProcessed])

    reconstructPCs = inferenceBatchReconstructAutoencoder(model, pcProcessed)

    for reconstructPC in reconstructPCs:
        savePath = os.path.join(saveFolder, instanceName)
        savePCAsObj(reconstructPC, savePath)


def main():
    #%% Load PointNet model
    modelPath = 'log/reconstruction/no_data_aug/checkpoints/best_model.pth'
    print("Model loaded.")
    # Initialize the model and load the trained weights
    model = get_model(2048, -1, False)
    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    #%% ShapeNet Dataset PC
    category = "02691156"
    pcDataPath = os.path.join(PROJECT_PATH, "generated", "ShapeNet", "ShapeNetCore_v2_PC", category)

    selectedNamePath = os.path.join(PROJECT_PATH, "generated", "Spaghetti", "plane_names_all_filtered.txt")
    selectedWholeNames = readSelectedMD5s_part(selectedNamePath)

    #%%
    saveFolderName = "shapeNet_reconstruct_autoencoder"
    saveFolder = os.path.join(PROJECT_PATH, "generated", "ShapeNet", saveFolderName, category)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    #%% Generate and save PointNet features
    instanceNames = [f for f in os.listdir(pcDataPath) if os.path.isfile(os.path.join(pcDataPath, f))]
    instanceNames = tqdm(instanceNames)
    for instanceName in instanceNames:
        if instanceName not in instanceNames:
            continue
        
        instanceNames.set_description(f"Start reconstruction for shape {instanceName}")
        reconstruction(model, instanceName, pcDataPath, saveFolder)


if __name__ == '__main__':
    main()