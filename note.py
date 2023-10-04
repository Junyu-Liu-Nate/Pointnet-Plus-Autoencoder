import numpy as np
import os

def main():
    dataFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    objectName = "Pointnet_Wholes"
    dataName = "airplane"
    dataPath = os.path.join(dataFolder, objectName, dataName, "1.npy")

    print(np.load(dataPath))


if __name__ == '__main__':
    main()