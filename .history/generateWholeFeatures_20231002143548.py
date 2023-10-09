import random
import numpy as np

from customized_inference import preProcessPC, inferenceBatch
from geometry import sampleFromMesh, fpsSample, is_inside_sphere

def computeWholeFeature(wholeVertices, numParts, minPoints = 1024):
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
        
        while len(partVertices) < 1024:
            radius += 0.1
            partVertices = []
            for i in range(len(wholeVertices)):
                if is_inside_sphere(wholeVertices[i], center, radius):
                    partVertices.append(wholeVertices[i])

        pcProcessed = preProcessPC(partVertices)
        
        partPCs.append(partVertices)

    partPCs = np.array(partPCs)
    print(partPCs.shape)







def main():
    pass

if __name__ == '__main__':
    is_generate = True
    main(is_generate)