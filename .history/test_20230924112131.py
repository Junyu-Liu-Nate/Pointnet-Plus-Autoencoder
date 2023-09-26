import torch
import numpy as np

from customized_inference import read_off, preProcessPC

def main():
    data_path = '/Users/liujunyu/Data/Research/BVC/ITSS/ModelNet40/airplane/train/'
    mesh_names = range(1,5)
    mesh_paths = []
    pcList = []
    for mesh_name in mesh_names:
        meshPath = data_path + 'airplane_' + f"{mesh_name:04d}" + '.off'
        mesh_paths.append(meshPath)

        pc = read_off(meshPath)
        pcProcessed = preProcessPC(pc)
        pcList.append(pcProcessed)
    pcArray = np.array(pcList)

    pcTensor = torch.tensor(pcArray, dtype=torch.float32)
    # pcTensor = torch.tensor(pcArray, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
    print(pcTensor.shape)

if __name__ == '__main__':
    main()