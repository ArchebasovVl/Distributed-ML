import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def train():
    torch.manual_seed(0)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)

    os.environ['USE_LIBUV'] = '0'
    init_process_group(backend='gloo')

    model = nn.Sequential(
        nn.Conv2d(1, 2, 5),
        nn.Conv2d(2, 2, 3),
        nn.MaxPool2d(3),
        nn.Flatten(),
        nn.Linear(98, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=3,
                            sampler=DistributedSampler(dataset)
                            )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model = DDP(model)

    for epoch in range(3):
        print(end=f'Epoch {epoch} ')
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print('Loss:', loss.item())

    destroy_process_group()

if __name__ == '__main__':
    train()