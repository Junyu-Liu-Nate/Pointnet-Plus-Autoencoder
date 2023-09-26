import torch
import numpy as np
# from pointnet_pytorch.pointnet.model import PointNetCls
# from models.pointnet2_cls_msg import get_model
from models.pointnet2_cls_ssg import get_model

def read_off(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if 'OFF' != lines[0].strip():
        raise('Not a valid OFF header')
    
    n_vertices, _, _ = tuple(map(int, lines[1].strip().split()))
    vertices = np.array([[float(s) for s in line.strip().split()] for line in lines[2:2+n_vertices]])
    
    return vertices

def write_features_to_txt(feature_list, filename):
    with open(filename, 'w') as f:
        for mesh_path, feature in feature_list:
            feature_str = " ".join(map(str, feature))
            f.write(f"{mesh_path} {feature_str}\n")

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

def inferenceFeature(meshFilePath):
    # modelPath = 'pointnet_pytorch/utils/cls/cls_model_75.pth'
    modelPath = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'

    # Initialize the model and load the trained weights
    # model = PointNetCls(k=5)  # Initialize with number of classes
    model = get_model(5, False)
    model.load_state_dict(torch.load(modelPath,  map_location=torch.device('cpu')))

    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)

    model.eval()

    # Read .off file and prepare the point cloud
    point_cloud = read_off(meshFilePath)
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)  # Convert to tensor and adjust dimensions

    # Perform inference
    with torch.no_grad():
        output, trans, trans_feat = model(point_cloud, True)
        # print(output, trans, trans_feat)

    return output.numpy()

def main():
    data_path = '/users/ljunyu/data/ljunyu/data/ModelNet40/airplane/train/'
    mesh_names = range(1,2)
    mesh_paths = []
    for mesh_name in mesh_names:
        mesh_paths.append(data_path + 'airplane_' + f"{mesh_name:04d}" + '.off')

    feature_list = []
    for mesh_path in mesh_paths:
        mesh_feature = inferenceFeature(mesh_path)[0]
        # print(mesh_feature)
        feature_list.append((mesh_path, mesh_feature))

    featureDataPath = "/users/ljunyu/data/ljunyu/data/" + "airplane_features.txt"
    # Writing features to txt
    write_features_to_txt(feature_list, featureDataPath)

    # Reading features from txt
    read_features = read_features_from_txt(featureDataPath)
    print("Read features:", read_features[0])

if __name__ == '__main__':
    main()

