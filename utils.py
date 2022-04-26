from torchvision import datasets
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
dataset_path = '/cs_storage/public_datasets/kinetics400/val'
datasets_names = ['kinetics400']
kinetics_id_to_classname = None

def parse():
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--model", "-m", choices=models_names, default='custom',
                        help="Run only specific model")
    parser.add_argument("--dataset", "-da", choices=datasets_names, default='cifar10',
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
    parser.add_argument("--frames", "-f", type=int, default=35,
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

    return model, dataset, eps, n_pop, n_gen, n_videos, tournament, n_frames, n_iter

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
    # tranposed_x = x.permute(0, 2, 1, 3, 4)
    squeezed_x = x.squeeze(dim=0)
    normalized_img = transform(squeezed_x).unsqueeze(dim=0)
    # normalized_img = normalized_img.permute(0, 2, 1, 3, 4)
    return normalized_img

def correctly_classified(model, x, y):
    normalized_x = normalize(x)
    preds = model(normalized_x)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print("########################################")
    print("Top names: %s" % ", ".join(pred_class_names))
    print(f'Top labels {np.array2string(np.array(pred_classes.cpu()))}')
    print(f'Original name: {kinetics_id_to_classname[int(y)]}')
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

def save_video(video, fname):
    p_video = video.squeeze(dim=0)
    p_video = p_video * 255.0
    p_video = p_video.to(torch.uint8)
    p_video = p_video.permute(1, 2, 3, 0)
    write_video(fname, p_video.cpu(), fps=10)


def get_dataset(n_videos, batch_size=1):
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
                label = classnames_to_id[f'{video_name}']
                label = torch.tensor(label)
                videos.append([inputs, label])
                break

        val_loader = DataLoader(videos, batch_size=batch_size, shuffle=True)

        return val_loader, videos