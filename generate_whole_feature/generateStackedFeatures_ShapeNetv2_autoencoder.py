import random
import numpy as np
import torch
import os
from tqdm import tqdm

# from models.pointnet2_cls_ssg_original import get_model
from models.pointnet2_cls_ssg import get_model

from collections import OrderedDict
from train_test_custom.customized_inference import preProcessPC, preProcessPC_nonormalize, inferenceBatchAutoencoder
from fileIO import read_obj_vertices

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

PARAMETERS = {
    "pc_folder": "ShapeNetv2_Wholes_sphere_50_r04_normalized_PC",
    "category": "02691156",

    "output_name": "ShapeNetv2_stacked_sphere_50_r04_autoencoder",
}

def main():
    ### Load model
    modelPath = 'log/reconstruction/shapeNet_FPS_airplane_parts_no_normalize/checkpoints/best_model.pth'
    print("Model loaded.")
    # Initialize the model and load the trained weights
    model = get_model(2048, -1, False)
    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    # Adjusting keys - For model trained with data parallel
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
        new_state_dict[name] = v
    # Loading the adjusted state dict
    model.load_state_dict(new_state_dict)
    model.eval()

    ### For ShapeNet
    pcDatasetPath = os.path.join(PROJECT_PATH, "generated", "ShapeNet", PARAMETERS["pc_folder"], PARAMETERS["category"])
    instance_names = [d for d in os.listdir(pcDatasetPath) if os.path.isdir(os.path.join(pcDatasetPath, d))]
    instance_names = tqdm(instance_names)

    save_folder = os.path.join(PROJECT_PATH, "generated", "ShapeNet", PARAMETERS["output_name"], PARAMETERS["category"])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for instance_name in instance_names:
        instance_names.set_description("Processing %s" % instance_name)

        partPCs = []
        for i in range(50):
            partPath = os.path.join(pcDatasetPath, instance_name, str(i) + ".obj")
            partVertices = read_obj_vertices(partPath)
            partVertices = np.array(partVertices)
            partProcessed = preProcessPC_nonormalize(2048, partVertices)
            partPCs.append(partProcessed)
        partPCs = np.array(partPCs)

        featureArray = inferenceBatchAutoencoder(model, partPCs)
        savePath = os.path.join(os.path.join(save_folder), str(instance_name) + ".npy")
        np.save(savePath, featureArray)


if __name__ == '__main__':
    main()