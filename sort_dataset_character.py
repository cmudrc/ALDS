from dataset.MatDataset import BurgersDataset
import torch
from models.teecnet import TEECNetConv
import matplotlib.pyplot as plt
import tqdm

model = TEECNetConv(1, 32, 1, num_layers=6, retrieve_weights=False, num_powers=3, sub_size=9)
dataset = BurgersDataset(root='data/burgers')

fft_x_list = []
fft_y_list = []

sub_x_total = []
sub_y_total = []

for data in tqdm.tqdm(dataset):
    data[0] = torch.tensor(data[0], dtype=torch.float)
    data[1] = torch.tensor(data[1], dtype=torch.float)
    sub_x_list, sub_y_list = model.get_partition_domain(data[0], mode='train'), model.get_partition_domain(data[1], mode='test')
    sub_x_total.append(sub_x_list)
    sub_y_total.append(sub_y_list)
    for sub_x, sub_y in zip(sub_x_list, sub_y_list):
        # calculate fft of sub_x and sub_y
        sub_x_fft = torch.fft.fftn(sub_x)
        sub_y_fft = torch.fft.fftn(sub_y)

        # calculate the dominant frequencies
        sub_x_fft_abs = torch.abs(sub_x_fft)
        sub_y_fft_abs = torch.abs(sub_y_fft)
        sub_x_fft_abs_max = torch.max(sub_x_fft_abs)
        sub_y_fft_abs_max = torch.max(sub_y_fft_abs)

        # store the dominant frequencies
        sub_x_fft_abs_max = sub_x_fft_abs_max.item()
        sub_y_fft_abs_max = sub_y_fft_abs_max.item()
        fft_x_list.append(sub_x_fft_abs_max)
        fft_y_list.append(sub_y_fft_abs_max)


# save the sub_x_total and sub_y_total lists and the fft_x_list and fft_y_list lists
torch.save(sub_x_total, 'sub_x_total.pt')
torch.save(sub_y_total, 'sub_y_total.pt')
torch.save(fft_x_list, 'fft_x_list.pt')
torch.save(fft_y_list, 'fft_y_list.pt')

# plot the distribution of the dominant frequencies of the input and output
plt.hist(fft_x_list, bins=100, alpha=0.5, label='input')
plt.hist(fft_y_list, bins=100, alpha=0.5, label='output')



        