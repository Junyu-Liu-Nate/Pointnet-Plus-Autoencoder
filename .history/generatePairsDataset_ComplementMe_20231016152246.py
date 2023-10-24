import os
import numpy as np
import random

from generateWholeFeatures import computeWholeFeature
from fileIO import read_features_from_txt, create_initial_txt, add_pair_to_existing_txt, read_features_from_txt_shapenet

BASE_DATA_PATH = "/Users/liujunyu/Data/Research/BVC/ITSS/"

def generatePairsDataset(outputPath):
    