from dataset.GraphDataset import CoronaryArteryDataset, DuctAnalysisDataset

if __name__ == '__main__':
    # dataset = CoronaryArteryDataset(root='data/coronary', partition=True, sub_size=5)
    dataset = DuctAnalysisDataset(root='data/Duct', partition=True, sub_size=0.05)
    print(len(dataset))
    # print(dataset[0])
    # print(len(dataset[0]))