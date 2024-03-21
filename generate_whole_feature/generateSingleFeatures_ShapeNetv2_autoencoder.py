import random
import numpy as np
import torch
import os
from tqdm import tqdm

# from models.pointnet2_cls_ssg_original import get_model
from models.pointnet2_cls_ssg import get_model

from collections import OrderedDict
from train_test_custom.customized_inference import preProcessPC, preProcessPC_nonormalize, inferenceBatchAutoencoder
from fileIO import read_obj_vertices, readSelectedMD5s_part

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

PARAMETERS = {
    "pc_folder": "ShapeNetCore_v2_PC",
    "category": "02691156",

    "queryNamesFile": "plane_names_all.txt"
    # "output_name": "ShapeNetv2_stacked_ellipsoid_100_autoencoder",
}

def write_dict_to_file(filename_dict, file_path):
    with open(file_path, 'w') as file:
        for key, array in filename_dict.items():
            # Convert the array to a string and remove brackets
            array_str = ' '.join(map(str, array))
            # Write the key and array to the file
            file.write(f"{key} {array_str}\n")

def main():
    ### Load model
    modelPath = 'log/reconstruction/no_data_aug/checkpoints/best_model.pth'
    print("Model loaded.")
    # Initialize the model and load the trained weights
    model = get_model(2048, -1, False)
    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    ### For ShapeNet
    pcDatasetPath = os.path.join(PROJECT_PATH, "generated", "ShapeNet", PARAMETERS["pc_folder"], PARAMETERS["category"])
    selectedNamePath = os.path.join(PROJECT_PATH, "generated", "Spaghetti", PARAMETERS["queryNamesFile"])
    instance_names = readSelectedMD5s_part(selectedNamePath)
    # instance_names = ["1a32f10b20170883663e90eaf6b4ca52"]
    instance_names = tqdm(instance_names)

    whole_features_dict = {}
    for instance_name in instance_names:
        instance_names.set_description("Processing %s" % instance_name)

        pc = read_obj_vertices(os.path.join(pcDatasetPath, instance_name + ".obj"))
        pcProcessed = preProcessPC_nonormalize(2048, pc)
        featureArray = inferenceBatchAutoencoder(model, np.array([pcProcessed]))
        whole_features_dict[instance_name] = featureArray[0]
    
    save_path = os.path.join(PROJECT_PATH, "generated", "ShapeNet", "autoencoder_airplane_ShapeNetv2" + ".txt")
    write_dict_to_file(whole_features_dict, save_path)


if __name__ == '__main__':
    main()