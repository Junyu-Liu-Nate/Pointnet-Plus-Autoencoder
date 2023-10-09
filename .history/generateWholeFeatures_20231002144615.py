import random
import numpy as np
import os

from customized_inference import preProcessPC, inferenceBatch
from geometry import sampleFromMesh, fpsSample, is_inside_sphere

def computeWholeFeature(model, wholeVertices, numParts, minPoints = 1024):
    """
        Compute feature for 1 whole shape: composed of numParts part features (numParts, 1024)
    """
    partPCs = []
    
    min_radius = 0.1
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

        partProcessed = preProcessPC(partVertices)
        partPCs.append(partProcessed)

    partPCs = np.array(partPCs)
    print(partPCs.shape)

    featureArray = inferenceBatch(model, partPCs)

    return featureArray


def main():
    dataPath = '/Users/liujunyu/Data/Research/BVC/ITSS/dataset/ModelNet40/airplane/train/'
    meshName = 'airplane_0001'
    meshPath = os.path.join(dataPath, meshName + ".off")

    #%% Load model
    # modelPath = 'pointnet_pytorch/utils/cls/cls_model_75.pth'
    modelPath = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    print("Model loaded.")

    # Initialize the model and load the trained weights
    model = get_model(40, False)

    loaded_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model_state_dict = loaded_dict['model_state_dict']
    model.load_state_dict(model_state_dict)

    model.eval()

    numSample = 16000
    wholeVertices = sampleFromMesh(meshPath, 16000)

    numParts = 50
    wholeFeature = computeWholeFeature(model, wholeVertices, numParts, 1024)


if __name__ == '__main__':
    is_generate = True
    main(is_generate)