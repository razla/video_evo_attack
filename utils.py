from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
import argparse

models_names = ['resnet_3d', 'resnet_mc', 'resnet_1d']
datasets_names = ['kinetics400']

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
    args = parser.parse_args()

    n_videos = args.videos
    dataset = args.dataset
    model = args.model
    tournament = args.tournament
    eps = args.eps
    n_pop = args.pop
    n_gen = args.gen
    n_iter = n_gen * n_pop

    return model, dataset, eps, n_pop, n_gen, n_videos, tournament, n_iter

def get_model(model_name):
    if model_name == 'resnet_3d':
        return r3d_18(pretrained=True)
    elif model_name == 'resnet_mc':
        return mc3_18(pretrained=True)
    elif model_name == 'resnet_1d':
        return r2plus1d_18(pretrained=True)
    else:
        raise Exception('No such model!')

def get_dataset(batch_size=64):
        test_set = datasets.Kinetics(root='./data',
                                      frames_per_clip=3,
                                      num_classes='400',
                                      download=False,
                                      split='val')

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        return test_loader