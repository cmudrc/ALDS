from dataset.GraphDataset import CoronaryArteryDataset

if __name__ == '__main__':
    dataset = CoronaryArteryDataset(root='data/coronary', partition=True, sub_size=5)