from dataset.GraphDataset import CoronaryArteryDataset

dataset = CoronaryArteryDataset(root='data/coronary', partition=True, sub_size = 100)

print(dataset[0])