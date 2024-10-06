from dataset.GraphDataset import CoronaryArteryDataset

<<<<<<< HEAD
if __name__ == '__main__':
    dataset = CoronaryArteryDataset(root='data/coronary', partition=True, sub_size=5)
=======
dataset = CoronaryArteryDataset(root='data/coronary', partition=True, sub_size = 100)

print(dataset[0])
>>>>>>> 5ffafbb975eafbaa3316789034833c8446b3d68c
