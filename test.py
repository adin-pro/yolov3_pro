from utils import *
import torch.utils.data


dataset = ListDataset('img/valid.txt', img_size=416,augment=True,multiscale=True)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=dataset.collate_fn
)

print(dataset[0][1][0])