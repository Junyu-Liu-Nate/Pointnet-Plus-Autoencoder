import random
import numpy as np
import torch
import os
from tqdm import tqdm

from models.pointnet2_cls_ssg import get_model
from data_utils.ModelNetDataLoader import pc_normalize, farthest_point_sample
from customized_inference import preProcessPC, inferenceBatch
from geometry import sampleFromMesh, fpsSample, is_inside_sphere, scale_to_unit_sphere
from fileIO import savePCAsObj, saveWholeFeature, read_obj_vertices

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

def main():
    #%% Load model
    modelPath = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    print("Model loaded.")
    # Initialize the model and load the trained weights
    model = get_model(40, False)
    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    #%% Note: read directly from pre-generated pc
    ### For ModelNet 40
    # pcDatasetPath = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/ModelNet40_PC/airplane/train/"
    # pcPaths = []
    # for pcName in range(1, 601):
    #     pcPath = os.path.join(pcDatasetPath, 'airplane_' + f"{pcName:04d}" + '.obj')
    #     pcPaths.append(pcPath)

    ### For ShapeNet
    pcDatasetPath = os.path.join(PROJECT_PATH, "generated", "ShapeNet", "ShapeNetCore_v2_PC", "02691156")
    pcNames = [f for f in os.listdir(pcDatasetPath) if os.path.isfile(os.path.join(pcDatasetPath, f))]
    pcPaths = []
    for pcName in pcNames:
        pcPath = os.path.join(pcDatasetPath, pcName)
        pcPaths.append(pcPath)

    outputFolder = os.path.join(PROJECT_PATH, "generated", "ShapeNet")
    outputName = "ShapeNetv2_Wholes_ellipsoid_100"
    objectName = "03001627"
    outputPath = os.path.join(outputFolder, outputName, objectName)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    # for i in range(len(meshPaths)):
    for i in range(len(pcPaths)):
        savePath = os.path.join(outputPath, pcNames[i][:-4])
        if os.path.exists(savePath):
            continue
        
        wholeVertices = read_obj_vertices(pcPaths[i])

        numParts = 100
        wholeFeature = computeWholeFeature(outputName, pcNames[i][:-4], model, wholeVertices, numParts, 1024)

        # savePath = os.path.join(outputPath, pcNames[i][:-4])
        saveWholeFeature(wholeFeature, savePath)

        print(f"Finish generating whole feature for idx: {pcNames[i][:-4]}.")


if __name__ == '__main__':
    main()