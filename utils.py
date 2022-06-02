from torch.utils.data import DataLoader
import torch
from argparse import ArgumentParser
import random
import os
import numpy as np

from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video


random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_names = ['resnet_3d', 'resnet_mc', 'resnet_1d', 'slowfast', 'i3d_ucf']
kinetics_id_to_classname = None
ucf101_id_to_classname = None
hmdb51_id_to_classname = None
parent_dir = None
datasets_names = ['kinetics400', 'ucf101', 'hmdb51']

def parse_main():
    global parent_dir
    parser = ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--model", "-m", choices=models_names, default='custom',
                        help="Run only specific model")
    parser.add_argument("--dataset", "-da", choices=datasets_names, default='kinetics400',
                        help="Run only specific dataset")
    parser.add_argument("--eps", "-ep", type=float, default=0.1,
                        help="Constrained optimization problem - epsilon")
    parser.add_argument("--k", "-k", type=int, default=5,
                        help="Top k prediction")
    parser.add_argument("--pop", "-pop", type=int, default=20,
                        help="Population size")
    parser.add_argument("--gen", "-g", type=int, default=1000,
                        help="Number of generations")
    parser.add_argument("--videos", "-v", type=int, default=3,
                        help="Maximal number of videos from dataset")
    parser.add_argument("--tournament", "-t", type=int, default=35,
                        help="Tournament selection")
    args = parser.parse_args()

    n_videos = args.videos
    dataset = args.dataset
    model = args.model
    tournament = args.tournament
    eps = args.eps
    top_k = args.k
    n_pop = args.pop
    n_gen = args.gen
    n_iter = n_gen * n_pop

    parent_dir = f'/home/razla/VideoAttack/scripts/{model}/results'

    return model, dataset, eps, top_k, n_pop, n_gen, n_videos, tournament, n_iter

def parse_finetune():
    parser = ArgumentParser(
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

def normalize(x):
    # transform = torchvision.transforms.Compose([
    #     NormalizeVideo(
    #         mean=[0.43216, 0.394666, 0.37645],
    #         std = [0.22803, 0.22145, 0.216989]
    #     ),
    # ])

    transform = transforms.Compose([
        video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    squeezed_x = x.squeeze(dim=0)
    np_x = squeezed_x.cpu().numpy()
    np_x = np.transpose(np_x, axes=(1, 0, 2, 3))
    # normalized_img = transform(np_x).unsqueeze(dim=0)
    normalized_img = transform(np_x)
    return normalized_img

def print_success(model, queries, x, y):
    normalized_x = normalize(x)
    preds = model(normalized_x)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print("########################################")
    print(f'Success!')
    print("Top labels: %s" % "\n\t\t\t".join(pred_class_names))
    print(f'Original name: {kinetics_id_to_classname[int(y)]}')
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
    print("Top labels: %s" % "\n\t\t\t".join(pred_class_names))
    print(f'Original name: {kinetics_id_to_classname[int(y)]}')
    print(f'Queries: {queries}')
    print("########################################")

def print_summary(dataset, model_name, n_videos, n_pop, n_gen, n_tournament, eps, asr, evo_queries):
    print('########################################')
    print(f'Summary:')
    print(f'\tDataset: {dataset}')
    print(f'\tModel: {model_name}')
    print(f'\tVideos: {n_videos}')
    print(f'\tPopulation: {n_pop}')
    print(f'\tGenerations: {n_gen}')
    print(f'\tTournament: {n_tournament}')
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