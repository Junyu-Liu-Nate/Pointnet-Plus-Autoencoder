import random
import numpy as np
import torch
import os

from models.pointnet2_cls_ssg import get_model

BASE_DATA_PATH = "/Users/liujunyu/Data/Research/BVC/ITSS/"

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

    #%% ComplementMe Dataset
    dataset = os.path.join(BASE_DATA_PATH, "dataset", "ComplementMe")
    datasetType = "components"
    category = "Airplane"
    dataType = "component_all_centered_point_clouds"
    pcDataPath = os.path.join(dataset, datasetType, category, dataType)

    instanceNames = [d for d in os.listdir(pcDataPath) if os.path.isdir(os.path.join(pcDataPath, d))]
    for instanceName in instanceNames:
        instanceFolder = os.path.join(pcDataPath, instanceName)
        part