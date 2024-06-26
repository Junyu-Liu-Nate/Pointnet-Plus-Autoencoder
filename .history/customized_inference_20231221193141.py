import torch
import numpy as np
import os

# from pointnet_pytorch.pointnet.model import PointNetCls
from models.pointnet2_cls_ssg import get_model
# from models.pointnet2_cls_msg import get_model
from data_utils.ModelNetDataLoader import pc_normalize, farthest_point_sample, pc_normalize_nonormalize
from fileIO import read_obj_vertices

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

#%% File I/O functions
def read_off(file_path):
    """
    Input: 
        file_path: path to .off file
    Output: 
        vertices: vertices, np.array, [npoint, D], D=3
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if 'OFF' != lines[0].strip():
        raise('Not a valid OFF header')
    
    n_vertices, _, _ = tuple(map(int, lines[1].strip().split()))
    vertices = np.array([[float(s) for s in line.strip().split()] for line in lines[2:2+n_vertices]])
    
    return vertices

# def write_features_to_txt(feature_list, filename):
#     with open(filename, 'w') as f:
#         for mesh_path, feature in feature_list:
#             feature_str = " ".join(map(str, feature))
#             f.write(f"{mesh_path} {feature_str}\n")

def write_features_to_txt(filename, feature_list, mesh_paths):
    with open(filename, 'w') as f:
        for i in range(len(feature_list)):
            feature_str = " ".join(map(str, feature_list[i]))
            f.write(f"{mesh_paths[i]} {feature_str}\n")

def read_features_from_txt(filename):
    mesh_features = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            mesh_path = parts[0]
            feature = np.array([float(x) for x in parts[1:]])
            mesh_features.append((mesh_path, feature))
    return [feature for _, feature in mesh_features]

#%% Data processing functions
def preProcessPC(pc):
    """
    Input: 
        pc: vertices
    Output:

    """
    npoint = 1024
    pcSampled = farthest_point_sample(pc, npoint)
    pcNormalized = pc_normalize(pcSampled)
    return pcNormalized

def preProcessPC_nonormalize(pc):
    """
    Input: 
        pc: vertices
    Output:

    """
    npoint = 1024
    pcSampled = farthest_point_sample(pc, npoint)
    pcNormalized = pc_normalize_nonormalize(pcSampled)
    return pcNormalized


#%% Model inference
# def inferenceFeature(model, meshFilePath):    
#     # Read .off file and prepare the point cloud
#     point_cloud = read_off(meshFilePath)
#     point_cloud = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)  # Convert to tensor and adjust dimensions

#     # Perform inference
#     with torch.no_grad():
#         output = model(point_cloud, True)
#         # print(output, trans, trans_feat)

#     return output.numpy()

def inferenceBatch(model, pcArray):   
    pcTensor = torch.tensor(pcArray, dtype=torch.float32).permute(0, 2, 1)

    # Perform inference
    with torch.no_grad():
        output = model(pcTensor, True)
        # print(output, trans, trans_feat)

    return output.numpy()

#%% Main
# def main():
#     #%% Load model
#     # modelPath = 'pointnet_pytorch/utils/cls/cls_model_75.pth'
#     modelPath = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
#     # modelPath = 'log/classification/pointnet2_msg_normals/checkpoints/best_model.pth'
#     print("Model loaded.")

#     # Initialize the model and load the trained weights
#     # # model = PointNetCls(k=5)  # Initialize with number of classes
#     model = get_model(40, False)
#     # model.load_state_dict(torch.load(modelPath,  map_location=torch.device('cpu')))

#     loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
#     model_state_dict = loaded_dict['model_state_dict']
#     model.load_state_dict(model_state_dict)

#     model.eval()

#     #%% Specify dataset
#     ### ModelNet 40 version
#     # data_path = '/Users/liujunyu/Data/Research/BVC/ITSS/ModelNet40/airplane/train/'
#     # mesh_names = range(1,601)
#     # mesh_paths = []
#     # pcList = []
#     # for mesh_name in mesh_names:
#     #     meshPath = data_path + 'airplane_' + f"{mesh_name:04d}" + '.off'
#     #     mesh_paths.append(meshPath)

#     #     pc = read_off(meshPath)
#     #     pcProcessed = preProcessPC(pc)
#     #     pcList.append(pcProcessed)
#     # pcArray = np.array(pcList)

#     ### ShapeNet v2 version
#     # data_path = '/Users/liujunyu/Data/Research/BVC/ITSS/dataset/ShapeNetCore_v2/02691156/'
#     data_path = os.path.join(DATASET_PATH, "ShapeNetCore_v2", "03001627")
#     meshNames = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
#     pcList = []
#     for meshName in meshNames:
#         meshPath = os.path.join(data_path, meshName, "models", "model_normalized.obj")

#         pc = read_obj_vertices(meshPath)
#         pcProcessed = preProcessPC(pc)
#         pcList.append(pcProcessed)
#     pcArray = np.array(pcList)

#     #%% Perform inference
#     # feature_list = []
#     # for mesh_path in mesh_paths:
#     #     mesh_feature = inferenceFeature(model, mesh_path)[0]
#     #     print(mesh_feature.shape)
#     #     feature_list.append((mesh_path, mesh_feature))
#     # features = inferenceFeature(model, mesh_paths)
#     # print(feature_list[0].shape)
#     featureArray = inferenceBatch(model, pcArray)
#     # print(featureArray)
#     featureList = featureArray.tolist()

#     #%% Write features to file
#     # featureDataPath = "/Users/liujunyu/Data/Research/BVC/ITSS/" + "pointnet2_chair_ShapeNetv2.txt"
#     featureDataPath = os.path.join(PROJECT_PATH, "generated", "ShapeNet")
#     # Writing features to txt
#     # write_features_to_txt(featureDataPath, featureList, mesh_paths)
#     write_features_to_txt(featureDataPath, featureList, meshNames) # Instead of writing the abosolute path, write the unique name

#     #%% Read features from file for inspection
#     # Reading features from txt
#     # read_features = read_features_from_txt(featureDataPath)
#     # print("Read features:", read_features[0])

def main():
    #%% Load model
    # modelPath = 'pointnet_pytorch/utils/cls/cls_model_75.pth'
    modelPath = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    # modelPath = 'log/classification/pointnet2_msg_normals/checkpoints/best_model.pth'
    print("Model loaded.")

    # Initialize the model and load the trained weights
    # # model = PointNetCls(k=5)  # Initialize with number of classes
    model = get_model(40, False)
    # model.load_state_dict(torch.load(modelPath,  map_location=torch.device('cpu')))

    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)

    model.eval()

    ### ShapeNet v2 version
    # data_path = '/Users/liujunyu/Data/Research/BVC/ITSS/dataset/ShapeNetCore_v2/02691156/'
    data_path = os.path.join(DATASET_PATH, "ShapeNetCore_v2", "03001627")
    meshNames = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    # Define batch size for processing
    batch_size = 1000  # Adjust this based on your memory capacity
    featureDataPath = os.path.join(PROJECT_PATH, "generated", "ShapeNet")

    # Open the file once and keep appending to it
    with open(featureDataPath, 'w') as feature_file:
        for i in range(0, len(meshNames), batch_size):
            # Process in batches
            batch_meshNames = meshNames[i:i + batch_size]
            batch_pcArray = np.array([preProcessPC(read_obj_vertices(os.path.join(data_path, meshName, "models", "model_normalized.obj"))) for meshName in batch_meshNames])
            
            # Perform inference
            batch_featureArray = inferenceBatch(model, batch_pcArray)
            batch_featureList = batch_featureArray.tolist()

            # Write features of this batch to file
            for meshName, feature in zip(batch_meshNames, batch_featureList):
                feature_str = " ".join(map(str, feature))
                feature_file.write(f"{meshName} {feature_str}\n")

            print(f"Processed batch {i//batch_size + 1}/{len(meshNames)//batch_size + 1}")

if __name__ == '__main__':
    main()

