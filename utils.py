from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
import torch
import argparse
import json
import urllib
import random
import os
import torchvision
from torchvision.io import write_video
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
import numpy as np
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_names = ['resnet_3d', 'resnet_mc', 'resnet_1d']
kinetics_dataset_path = '/cs_storage/public_datasets/kinetics400/val'
ucf101_dataset_path = '/cs_storage/public_datasets/UCF101/UCF-101'
ucf101_labels_path = '/cs_storage/public_datasets/UCF101/ucfTrainTestlist/classInd.txt'
hmdb51_dataset_path = '/cs_storage/public_datasets/HMDB51/HMDB-51'
kinetics_id_to_classname = None
ucf101_id_to_classname = None
hmdb51_id_to_classname = None
parent_dir = None
datasets_names = ['kinetics400', 'ucf101', 'hmdb51']

def parse_main():
    global parent_dir
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--model", "-m", choices=models_names, default='custom',
                        help="Run only specific model")
    parser.add_argument("--dataset", "-da", choices=datasets_names, default='kinetics400',
                        help="Run only specific dataset")
    parser.add_argument("--eps", "-ep", type=float, default=0.1,
                        help="Constrained optimization problem - epsilon")
    parser.add_argument("--pop", "-pop", type=int, default=20,
                        help="Population size")
    parser.add_argument("--gen", "-g", type=int, default=1000,
                        help="Number of generations")
    parser.add_argument("--videos", "-v", type=int, default=3,
                        help="Maximal number of videos from dataset")
    parser.add_argument("--tournament", "-t", type=int, default=35,
                        help="Tournament selection")
    parser.add_argument("--frames", "-f", type=int, default=5,
                        help="Number of frames to be perturbed in each mutation")
    args = parser.parse_args()

    n_videos = args.videos
    dataset = args.dataset
    model = args.model
    tournament = args.tournament
    n_frames = args.frames
    eps = args.eps
    n_pop = args.pop
    n_gen = args.gen
    n_iter = n_gen * n_pop

    parent_dir = f'/home/razla/VideoAttack/scripts/{model}/results'

    return model, dataset, eps, n_pop, n_gen, n_videos, tournament, n_frames, n_iter

def parse_finetune():
    parser = argparse.ArgumentParser(
        description="Finetune different models")
    parser.add_argument("--model", "-m", choices=models_names, default='custom',
                        help="Run only specific model")
    parser.add_argument("--dataset", "-d", choices=datasets_names, default='ucf101',
                        help="Run only specific dataset")
    parser.add_argument("--n_epochs", "-e", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=64,
                        help="Batch size")
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    return model, dataset, n_epochs, batch_size

def get_model(model_name):
    if model_name == 'resnet_3d':
        model = r3d_18(pretrained=True)
    elif model_name == 'resnet_mc':
        model = mc3_18(pretrained=True)
    elif model_name == 'resnet_1d':
        model = r2plus1d_18(pretrained=True)
    else:
        raise Exception('No such model!')
    model = model.eval()
    return model

def normalize(x):
    transform = torchvision.transforms.Compose([
        NormalizeVideo(
            mean=[0.43216, 0.394666, 0.37645],
            std = [0.22803, 0.22145, 0.216989]
        ),
    ])
    squeezed_x = x.squeeze(dim=0)
    normalized_img = transform(squeezed_x).unsqueeze(dim=0)
    return normalized_img

def correctly_classified(dataset, model, x, y):
    normalized_x = normalize(x)
    preds = model(normalized_x)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]
    if dataset == 'kinetics400':
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
        id_to_classname = kinetics_id_to_classname
    elif dataset == 'ucf101':
        pred_class_names = [ucf101_id_to_classname[int(i)] for i in pred_classes]
        id_to_classname = ucf101_id_to_classname
    print("########################################")
    print("Top names: %s" % ", ".join(pred_class_names))
    print(f'Top labels {np.array2string(np.array(pred_classes.cpu()))}')
    print(f'Original name: {id_to_classname[int(y)]}')
    print(f'Original class: {int(y)}')
    print("########################################")
    return y in pred_classes

def print_success(model, queries, x, y):
    normalized_x = normalize(x)
    preds = model(normalized_x)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print("########################################")
    print(f'Success!')
    print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))
    print(f'Original class: {kinetics_id_to_classname[int(y)]}')
    print(f'Queries: {queries}')
    print("########################################")

def print_failure(model, queries, x, y):
    normalized_x = normalize(x)
    preds = model(normalized_x)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print("########################################")
    print(f'Failure!')
    print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))
    print(f'Original class: {kinetics_id_to_classname[int(y)]}')
    print(f'Queries: {queries}')
    print("########################################")

def print_summary(dataset, model_name, n_videos, n_pop, n_gen, n_tournament, n_frames, eps, asr, evo_queries):
    print('########################################')
    print(f'Summary:')
    print(f'\tDataset: {dataset}')
    print(f'\tModel: {model_name}')
    print(f'\tVideos: {n_videos}')
    print(f'\tPopulation: {n_pop}')
    print(f'\tGenerations: {n_gen}')
    print(f'\tTournament: {n_tournament}')
    print(f'\tFrames: {n_frames}')
    print(f'\tMetric: linf, epsilon: {eps:.4f}')
    print(f'\tEvo:')
    print(f'\t\tEvo - attack success rate: {asr * 100:.4f}%')
    print(f'\t\tEvo - queries: {evo_queries}')
    print(f'\t\tEvo - queries (median): {int(np.median(evo_queries))}')
    print('########################################')

def save_video(x, y, fname):
    p_x = x.squeeze(dim=0)
    p_x = p_x * 255.0
    p_x = p_x.to(torch.uint8)
    p_x = p_x.permute(1, 2, 3, 0)
    dir_path = os.path.join(parent_dir, str(y.item()))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, fname)
    write_video(file_path, p_x.cpu(), fps=10)

def get_dataset(dataset, n_videos, batch_size=1):
    if dataset == 'kinetics400':
        dataset_path = kinetics_dataset_path
        json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
        json_filename = "kinetics_classnames.json"
        try:
            urllib.request.urlretrieve(json_url, json_filename)
        except:
            print('Error downloading json file')
        with open(json_filename, "r") as f:
            kinetics_classnames = json.load(f)
        global kinetics_id_to_classname

        kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            kinetics_id_to_classname[v] = str(k).replace('"', "")

        classnames_to_id = {}
        for k, v in kinetics_classnames.items():
            k = k.replace('"', "")
            k = k.replace('\'', "")
            k = k.replace('(', "")
            k = k.replace(')', "")
            classnames_to_id[k] = v

    elif dataset == 'ucf101':
        global ucf101_id_to_classname
        dataset_path = ucf101_dataset_path
        with open(ucf101_labels_path, "r") as f:
            ucf101_classnames = f.read().split('\n')
        classnames_to_id = {}
        ucf101_id_to_classname = {}
        for line in ucf101_classnames:
            id, classname = line.split(' ')
            classnames_to_id[classname] = int(id)
            ucf101_id_to_classname[id] = classname
        f.close()

    elif dataset == 'hmdb51':
        global hmdb51_id_to_classname
        dataset_path = hmdb51_dataset_path
        classnames_to_id = {}
        ucf101_id_to_classname = {}
        for id, classname in enumerate(os.listdir(hmdb51_dataset_path)):
            classname = classname.replace('_', ' ')
            classnames_to_id[classname] = int(id)
            ucf101_id_to_classname[id] = classname

    side_size = 112
    crop_size = 112
    num_frames = 16
    sampling_rate = 8
    frames_per_second = 30

    # Note that this transform is specific to the slow_R50 model.
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                # NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size))
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate) / frames_per_second
    start_sec = 0
    end_sec = start_sec + clip_duration
    videos = []

    videos_dirs = os.listdir(dataset_path)
    if n_videos > len(videos_dirs):
        videos_names = videos_dirs
    else:
        videos_names = random.choices(os.listdir(dataset_path), k=n_videos)
    for video_name in videos_names:
        for f in os.scandir(dataset_path + '/' + video_name):
            video = EncodedVideo.from_path(f)
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            video_data = transform(video_data)
            inputs = video_data["video"]
            video_name = video_name.replace('_', ' ')
            label = classnames_to_id[video_name]
            label = torch.tensor(label)
            videos.append([inputs, label])
            break

    val_loader = DataLoader(videos, batch_size=batch_size, shuffle=True)

    return val_loader, videos