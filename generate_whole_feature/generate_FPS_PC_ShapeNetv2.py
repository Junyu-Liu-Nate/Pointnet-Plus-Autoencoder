import random
import numpy as np
import torch
import os
from tqdm import tqdm

from models.pointnet2_cls_ssg_original import get_model
from data_utils.ModelNetDataLoader import pc_normalize, farthest_point_sample
from train_test_custom.customized_inference import preProcessPC, inferenceBatch
from geometry import sampleFromMesh, fpsSample, is_inside_sphere, scale_to_unit_sphere
from fileIO import savePCAsObj, saveWholeFeature, read_obj_vertices, get_names_from_json, save_parameters

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

PARAMETERS = {
    "pc_folder": "ShapeNetCore_v2_PC",
    "category": "02691156",

    "name_select_filename": "spaghetti_airplanes_train.json",
    "output_name": "ShapeNetv2_Wholes_sphere_50_r04_normalized",
    
    "num_parts": 50,
    "min_radius": 0.4,
    "max_radius": 0.4,

    "Notes": "Sphere cut. The whole shape is both center and scale normalized. But the parts are not. Each part sampled by 2048."
}

def is_inside_sphere(point, center, radius):
    """
    Check if a point is inside the given sphere.
    """
    return np.linalg.norm(point - center) <= radius

def is_inside_ellipsoid(point, center, radius_x, radius_y, radius_z):
    """
    Check if a point is inside the given ellipsoid.
    """
    x, y, z = point
    cx, cy, cz = center
    return ((x - cx)**2 / radius_x**2 + (y - cy)**2 / radius_y**2 + (z - cz)**2 / radius_z**2) <= 1

def fps_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def fps_cut(outputName, objectName, wholeIdx, wholeVertices, numParts, minPoints=1024):
    """
    Compute feature for 1 whole shape: composed of numParts part features (numParts, 1024)
    """

    min_radius = PARAMETERS["min_radius"]
    max_radius = PARAMETERS["max_radius"]

    centers = fps_sample(wholeVertices, numParts)

    for idx in range(numParts):
        # radius_x = radius_y = radius_z = random.uniform(min_radius, max_radius)
        radius_x = random.uniform(min_radius, max_radius)
        # radius_y = random.uniform(min_radius, max_radius)
        # radius_z = random.uniform(min_radius, max_radius)
        center = centers[idx]

        partVertices = []
        for i in range(len(wholeVertices)):
            # if is_inside_ellipsoid(wholeVertices[i], center, radius_x, radius_y, radius_z):
            #     partVertices.append(wholeVertices[i])
            if is_inside_sphere(wholeVertices[i], center, radius_x):
                partVertices.append(wholeVertices[i])
        
        while len(partVertices) < minPoints:
            radius_x += 0.1
            # radius_y += 0.05
            # radius_z += 0.05
            partVertices = []
            for i in range(len(wholeVertices)):
                # if is_inside_ellipsoid(wholeVertices[i], center, radius_x, radius_y, radius_z):
                #     partVertices.append(wholeVertices[i])
                if is_inside_sphere(wholeVertices[i], center, radius_x):
                    partVertices.append(wholeVertices[i])

        partVertices = np.array(partVertices)
        partVertices_fps = fps_sample(partVertices, 2048)

        visualizeBaseFolder = os.path.join(PROJECT_PATH, "generated", "ShapeNet", outputName + "_PC", objectName)
        visualizeFolder = os.path.join(visualizeBaseFolder, str(wholeIdx))
        if not os.path.exists(visualizeFolder):
            os.makedirs(visualizeFolder)
        parameter_file_path = os.path.join(visualizeBaseFolder, "parameters.txt")
        # Check if the file exists
        if not os.path.exists(parameter_file_path):
            # File does not exist, call save_parameters function
            save_parameters(PARAMETERS, parameter_file_path)

        visualizePath = os.path.join(visualizeFolder, str(idx) + ".obj")
        savePCAsObj(partVertices_fps, visualizePath) # This is non-normalized, which preserves the original PC positions


def main():
    ### For ShapeNet
    pcDatasetPath = os.path.join(PROJECT_PATH, "generated", "ShapeNet", PARAMETERS["pc_folder"], PARAMETERS["category"])
    spaghettiNamePath = os.path.join(PROJECT_PATH, "generated", "Spaghetti", PARAMETERS["name_select_filename"])
    pcNamesPure = get_names_from_json(spaghettiNamePath, PARAMETERS["category"])
    pcNames = []
    for pcNamePure in pcNamesPure:
        pcNames.append(pcNamePure + ".obj")
    print(len(pcNames))
    
    pcPaths = []
    for pcName in pcNames:
        pcPath = os.path.join(pcDatasetPath, pcName)
        pcPaths.append(pcPath)

    outputFolder = os.path.join(PROJECT_PATH, "generated", "ShapeNet")
    outputName = PARAMETERS["output_name"]
    objectName = "02691156"
    outputPath = os.path.join(outputFolder, outputName, objectName)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    save_parameters(PARAMETERS, os.path.join(outputPath, "parameters.txt"))

    print(len(pcPaths))

    for i in range(len(pcPaths)):
        savePath = os.path.join(outputPath, pcNames[i][:-4])
        check_path = os.path.join(outputFolder, outputName + "_PC", objectName, pcNames[i][:-4])
        if os.path.exists(check_path):
            print(f"[{i}/{len(pcPaths)}] Already generated whole feature for idx: {pcNames[i][:-4]}.")
            continue
        
        wholeVertices = read_obj_vertices(pcPaths[i])
        whole_vertices_normalized = pc_normalize(wholeVertices)

        numParts = PARAMETERS["num_parts"]
        fps_cut(outputName, objectName, pcNames[i][:-4], whole_vertices_normalized, numParts, 2048)

        print(f"[{i}/{len(pcPaths)}] Finish generating whole feature for idx: {pcNames[i][:-4]}.")


if __name__ == '__main__':
    main()