from generateWholeFeatures import computeWholeFeature

def generatePairsDataset(outputPath, wholeMeshPaths):
    datasetFolder = "/Users/liujunyu/Data/Research/BVC/ITSS/dataset/"
    poinrnetWholeFoler = "Pointnet_Wholes"
    poinrnetWholeObject = "airplane"

    for wholeMeshPath in wholeMeshPaths:
        meshIdx = int(wholeMeshPath[-8:-4])
        print(f"meshIdx: {meshIdx}")
        



def main():
    pass

if __name__ == '__main__':
    main()