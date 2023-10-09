import numpy as np
import os

def main():
    dataFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    dataName = "Pointnet_Wholes"
    dataName = "airplane"
    dataPath = os.path.join(dataFolder, dataName, dataName)

    print(np.load(file_path))


if __name__ == '__main__':
    main()