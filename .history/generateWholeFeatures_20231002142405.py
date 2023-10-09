import random

from customized_inference import preProcessPC, inferenceBatch
from geometry import sampleFromMesh, fpsSample

def computeWholeFeature(wholeVertices, numParts, minPoints = 1024):
    partPCs = []
    
    min_radius = 0.1
    max_radius = 0.3

    centers = fpsSample(wholeVertices, numParts)

    for idx in range(numParts):
        radius = random.uniform(min_radius, max_radius)
        center = centers[idx]

        part_vertices = []
        for i in range(len(wholeVertices)):
            if 




def main():
    pass

if __name__ == '__main__':
    is_generate = True
    main(is_generate)