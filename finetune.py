from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101
from torchvision import transforms
import random
import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_model, parse_finetune

ucf_data_dir = '/cs_storage/public_datasets/UCF101/UCF-101'
ucf_label_dir = '/cs_storage/public_datasets/UCF101/ucfTrainTestlist'
hmdb_data_dir = '/cs_storage/public_datasets/HMDB51'
model_path = '/home/razla/VideoAttack/scripts/UCF101'
frames_per_clip = 16
step_between_clips = 1
batch_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mean = (0.43216, 0.394666, 0.37645)
std = (0.22803, 0.22145, 0.216989)
random.seed(1)

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def get_dataset(ds_name, batch_size=512):
    if ds_name == 'ucf101':
        tfs = transforms.Compose([
            # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (T, C, H, W) for easier convolutions
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            # rescale to the most common size
            transforms.Lambda(lambda x: nn.functional.interpolate(x, (112, 112))),
            # Normalize video according to mean and std of pretrained model
            transforms.Normalize(mean, std),
        ])

        test_data = UCF101(root=ucf_data_dir,
                           annotation_path=ucf_label_dir,
                           frames_per_clip=frames_per_clip,
                           step_between_clips=step_between_clips,
                           frame_rate=32,
                           fold=1,
                           train=False,
                           transform=tfs, )

        train_data = UCF101(root=ucf_data_dir,
                      annotation_path=ucf_label_dir,
                      frames_per_clip=frames_per_clip,
                      step_between_clips=step_between_clips,
                      frame_rate=32,
                      fold=1,
                      train=True,
                      transform=tfs, )



    elif ds_name == 'hmdb51':
        raise Exception('No hmdb51!')

    # t_data = torch.permute(t_data, (1, 2, 0, 3, 4))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, collate_fn=custom_collate)

    return train_loader, test_loader

def train(train_loader, model, num_epochs):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_accuracy = 0

    # Train Network
    for epoch in range(num_epochs):
        losses = []

        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            data = torch.permute(data, (0, 2, 1, 3, 4))
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        accuracy = check_accuracy(test_loader, model)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, model_path)

        print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            data = torch.permute(data, (0, 2, 1, 3, 4))
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

    return float(num_correct)/float(num_samples)*100



if __name__ == '__main__':
    model_name, dataset, n_epochs, batch_size = parse_finetune()
    model = get_model(model_name)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(512, 101)
    )
    model = model.to(device)
    train_loader, test_loader = get_dataset(dataset, batch_size)
    train(train_loader, model, n_epochs)
    check_accuracy(test_loader, model)