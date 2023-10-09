import random
import numpy as np

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
    pass

if __name__ == '__main__':
    is_generate = True
    main(is_generate)