from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os

ucf_data_dir = '/cs_storage/public_datasets/UCF101/UCF-101'
ucf_label_dir = '/cs_storage/public_datasets/UCF101/ucfTrainTestlist'
hmdb_data_dir = '/cs_storage/public_datasets/HMDB51'
model_path = '/home/razla/VideoAttack/scripts/UCF101'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mean = (0.43216, 0.394666, 0.37645)
std = (0.22803, 0.22145, 0.216989)
random.seed(1)

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def get_dataset(ds_name, batch_size=32):

    dataset_path = ucf_data_dir

    f = open(os.path.join(ucf_label_dir, 'classInd.txt'), "r")
    classnames_and_ids = f.read()
    classnames_and_ids = classnames_and_ids.split('\n')
    classnames_to_id = {}
    for row in classnames_and_ids:
        id, cls = row.split(' ')
        classnames_to_id[cls] = int(id) - 1

    side_size = 112
    crop_size = 112
    num_frames = 16
    sampling_rate = 8
    frames_per_second = 30

    # Note that this transform is specific to the slow_R50 model.
    # Note that this transform is specific to the slow_R50 model.
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size))
            ]
        ),
    )

    f1 = open(os.path.join(ucf_label_dir, 'trainlist01.txt'), "r")
    # f2 = open(os.path.join(ucf_label_dir, 'trainlist02.txt'), "r")
    # f3 = open(os.path.join(ucf_label_dir, 'trainlist03.txt'), "r")
    trainlist01 = f1.read()
    # trainlist02 = f2.read()
    # trainlist03 = f3.read()
    spt_trainlist01 = trainlist01.split('\n')
    # spt_trainlist02 = trainlist02.split('\n')
    # spt_trainlist03 = trainlist03.split('\n')

    train_list = spt_trainlist01 # + spt_trainlist02 + spt_trainlist03

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate) / frames_per_second
    start_sec = 0
    end_sec = start_sec + clip_duration
    train_videos = []

    for row in train_list:
        video_path, label = row.split(' ')
        f = os.path.join(dataset_path, video_path)
        video = EncodedVideo.from_path(f)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = transform(video_data)
        inputs = video_data["video"]
        label = torch.tensor(int(label))
        train_videos.append([inputs, label])

    f1 = open(os.path.join(ucf_label_dir, 'testlist01.txt'), "r")
    # f2 = open(os.path.join(ucf_label_dir, 'testlist02.txt'), "r")
    # f3 = open(os.path.join(ucf_label_dir, 'testlist03.txt'), "r")
    testlist01 = f1.read()
    # testlist02 = f2.read()
    # testlist03 = f3.read()
    spt_testlist01 = testlist01.split('\n')
    # spt_testlist02 = testlist02.split('\n')
    # spt_testlist03 = testlist03.split('\n')

    test_list = spt_testlist01 # + spt_testlist02 + spt_testlist03

    test_videos = []

    for row in test_list:
        video_path, label = row.split(' ')
        f = os.path.join(dataset_path, video_path)
        video = EncodedVideo.from_path(f)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = transform(video_data)
        inputs = video_data["video"]
        label = torch.tensor(int(label))
        test_videos.append([inputs, label])

    train_loader = DataLoader(train_videos, batch_size=batch_size, shuffle=True)
    test_videos = DataLoader(test_videos, batch_size=batch_size, shuffle=True)

    return train_loader, test_videos

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
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    return float(num_correct)/float(num_samples)*100

    model.train()

if __name__ == '__main__':
    num_epochs = 30
    model = r3d_18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(512, 101)
    )
    model = model.to(device)
    train_loader, test_loader = get_dataset('ucf101')
    train(train_loader, model, num_epochs)
    check_accuracy(test_loader, model)
