import os
import numpy as np
import random

from generateWholeFeatures import computeWholeFeature
from fileIO import read_features_from_txt, create_initial_txt, add_pair_to_existing_txt, read_features_from_txt_shapenet

BASE_DATA_PATH = "/Users/liujunyu/Data/Research/BVC/ITSS/"

#%% ComplementMe Dataset
    dataset = os.path.join(BASE_DATA_PATH, "dataset", "ComplementMe")
    datasetType = "parts"
    category = "Airplane"
    dataType = "part_all_centered_point_clouds"
    pcDataPath = os.path.join(dataset, datasetType, category, dataType)