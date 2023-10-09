import random
import numpy as np
import torch
import os

from models.pointnet2_cls_ssg import get_model
from data_utils.ModelNetDataLoader import pc_normalize, farthest_point_sample
from customized_inference import preProcessPC, inferenceBatch
from geometry import sampleFromMesh, fpsSample, is_inside_sphere, scale_to_unit_sphere
from fileIO import savePCAsObj, saveWholeFeature

def computeWholeFeature(outputName, wholeIdx, model, wholeVertices, numParts, minPoints = 1024):
    """
        Compute feature for 1 whole shape: composed of numParts part features (numParts, 1024)
    """
    partPCs = []
    
    min_radius = 0.05
    max_radius = 0.3

    centers = fpsSample(wholeVertices, numParts)

    for idx in range(numParts):
        radius = random.uniform(min_radius, max_radius)
        center = centers[idx]

        partVertices = []
        for i in range(len(wholeVertices)):
            if is_inside_sphere(wholeVertices[i], center, radius):
                partVertices.append(wholeVertices[i])
        
        while len(partVertices) < minPoints:
            radius += 0.1
            partVertices = []
            for i in range(len(wholeVertices)):
                if is_inside_sphere(wholeVertices[i], center, radius):
                    partVertices.append(wholeVertices[i])

        partVertices = np.array(partVertices)
        partProcessed = preProcessPC(partVertices)
        partPCs.append(partProcessed)

        # visualizeBaseFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/visualize/Pointnet_Whole_enlarged200/"
        visualizeBaseFolder = os.path.join(
        visualizeFolder = os.path.join(visualizeBaseFolder, str(wholeIdx))
        if not os.path.exists(visualizeFolder):
            os.makedirs(visualizeFolder)
        visualizePath = os.path.join(visualizeFolder, str(idx) + ".obj")
        savePCAsObj(partProcessed, visualizePath)

    partPCs = np.array(partPCs)
    # print(partPCs)

    featureArray = inferenceBatch(model, partPCs)

    return featureArray


def main():
    #%%
    # dataPath = '/Users/liujunyu/Data/Research/BVC/ITSS/dataset/ModelNet40/airplane/train/'
    # meshName = 'airplane_0001'
    # meshPath = os.path.join(dataPath, meshName + ".off")

    #%% Load model
    modelPath = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    print("Model loaded.")
    # Initialize the model and load the trained weights
    model = get_model(40, False)
    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    # #%%
    # numSample = 16000
    # wholeVertices = sampleFromMesh(meshPath, numSample)

    # #%%
    # numParts = 50
    # wholeFeature = computeWholeFeature(model, wholeVertices, numParts, 1024)
    # # print(wholeFeature.shape)

    #%%
    datasetPath = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/ModelNet40/airplane/train/"
    meshNames = range(1,601)
    meshPaths = []
    for meshName in meshNames:
        meshPath = os.path.join(datasetPath, 'airplane_' + f"{meshName:04d}" + '.off')
        meshPaths.append(meshPath)

    outputFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    outputName = "Pointnet_Wholes_enlarged200"
    objectName = "airplane"
    outputPath = os.path.join(outputFolder, outputName, objectName)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for i in range(len(meshPaths)):
        numSample = 16000
        wholeVertices = sampleFromMesh(meshPaths[i], numSample)

        numParts = 200
        wholeFeature = computeWholeFeature(outputName, i, model, wholeVertices, numParts, 1024)

        savePath = os.path.join(outputPath, str(i))
        saveWholeFeature(wholeFeature, savePath)

        print(f"Finish generating whole feature for idx: {i}.")


if __name__ == '__main__':
    main()