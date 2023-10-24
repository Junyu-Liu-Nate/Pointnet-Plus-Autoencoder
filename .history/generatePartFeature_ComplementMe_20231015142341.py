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
    dataPath = os.path.join(BASE_DATA_PATH, "dataset", "ComplementMe")
    dataType = 