import random

from customized_inference import preProcessPC, inferenceBatch
from geometry import sampleFromMesh, fpsSample

def computeWholeFeature(pc, numParts, minPoints = 1024):
    partPCs = []
    
    min_radius = 0.1
    max_radius = 0.3

    centers = fpsSample(pc, numParts)

    for idx in range(numParts):
        radius = random.uniform(min_radius, max_radius)
        center = centers[i]

        for 



def main():
    pass

if __name__ == '__main__':
    is_generate = True
    main(is_generate)