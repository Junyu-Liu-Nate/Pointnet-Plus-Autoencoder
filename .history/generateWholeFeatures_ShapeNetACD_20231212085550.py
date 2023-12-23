import random
import numpy as np
import torch
import os
from tqdm import tqdm

from models.pointnet2_cls_ssg import get_model
from data_utils.ModelNetDataLoader import pc_normalize, farthest_point_sample
from customized_inference import preProcessPC, inferenceBatch
from geometry import sampleFromMesh, fpsSample, is_inside_sphere, scale_to_unit_sphere
from fileIO import savePCAsObj, saveWholeFeature, read_obj_vertices

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"