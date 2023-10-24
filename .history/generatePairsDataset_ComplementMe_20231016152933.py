import os
import numpy as np
import random

from generateWholeFeatures import computeWholeFeature
from fileIO import read_features_from_txt, create_initial_txt, add_pair_to_existing_txt, read_features_from_txt_shapenet

BASE_DATA_PATH = "/Users/liujunyu/Data/Research/BVC/ITSS/"

def generatePairsDataset(outputPath):
    #%% ComplementMe Dataset
    dataset = os.path.join(BASE_DATA_PATH, "dataset", "ComplementMe")
    datasetType = "parts"
    category = "Airplane"
    dataType = "part_all_centered_point_clouds"
    featureDataPath = os.path.join(dataset, datasetType, category, dataType)

    instanceNames = [d for d in os.listdir(featureDataPath) if os.path.isdir(os.path.join(featureDataPath, d))]

    for instanceName in instanceNames:
        instanceFolder = os.path.join(featureDataPath, instanceName)
        partNames = [f for f in os.listdir(instanceFolder) if os.path.isfile(os.path.join(instanceFolder, f))]

        for partName in partNames:
            partFeature = np.load()