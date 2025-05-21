import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        return rank, world_size, True
    return 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def train(rank, world_size, distributed):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

    if distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler, shuffle=(sampler is None), num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    if distributed:
        model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        if distributed:
            sampler.set_epoch(epoch)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        end_time = time.time()
        if not distributed or rank == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/total:.4f}, Accuracy = {100*correct/total:.2f}%, Time = {end_time - start_time:.2f}s")

def main():
    rank, world_size, distributed = setup_distributed()
    try:
        train(rank, world_size, distributed)
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()

