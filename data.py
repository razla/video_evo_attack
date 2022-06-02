from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from decord import VideoReader
import urllib
import random
import json
import os

KINETICS_DATASET_PATH = '/cs_storage/public_datasets/kinetics400/val'
KINETICS_LABELS_PATH = 'https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json'
UCF101_DATASET_PATH = '/cs_storage/public_datasets/UCF101/UCF-101'
UCF101_LABELS_PATH = '/cs_storage/public_datasets/UCF101/ucfTrainTestlist/classInd.txt'
HMDB51_DATASET_PATH = '/cs_storage/public_datasets/HMDB51/HMDB-51'

def get_dataset(dataset, n_videos):
    if dataset == 'kinetics400':
        dataset_path = KINETICS_DATASET_PATH
        json_url = KINETICS_LABELS_PATH
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
        dataset_path = UCF101_DATASET_PATH
        with open(UCF101_LABELS_PATH, "r") as f:
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
        dataset_path = HMDB51_DATASET_PATH
        classnames_to_id = {}
        ucf101_id_to_classname = {}
        for id, classname in enumerate(os.listdir(HMDB51_DATASET_PATH)):
            classname = classname.replace('_', ' ')
            classnames_to_id[classname] = int(id)
            ucf101_id_to_classname[id] = classname

    videos = []
    videos_dirs = os.listdir(dataset_path)
    if n_videos > len(videos_dirs):
        videos_names = videos_dirs
    else:
        videos_names = random.choices(os.listdir(dataset_path), k=n_videos)
    for video_name in videos_names:
        path = dataset_path + '/' + video_name
        for file in os.listdir(path):
            video = VideoReader(path + '/' + file)
            frame_id_list = range(0, 64, 2)
            video_frames = video.get_batch(frame_id_list).asnumpy()
            transform_fn = get_transform(dataset, normalize=False)
            t_video = transform_fn(video_frames)
            video_name = video_name.replace('_', ' ')
            label = classnames_to_id[video_name]
            videos.append([t_video, label])
            break

    return videos

def mean_std_values(dataset):
    if dataset == 'kinetics400':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset == 'ucf101':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset == 'hmdb51':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise Exception('No such dataset!')

def get_transform(dataset, normalize=True):
    mean, std = mean_std_values(dataset)
    if normalize:
        transform_fn = transforms.Compose([
            video.VideoCenterCrop(size=224),
            video.VideoToTensor(),
            video.VideoNormalize(mean=mean, std=std)
        ])
    else:
        transform_fn = transforms.Compose([
            video.VideoCenterCrop(size=224),
            video.VideoToTensor(),
        ])
    return transform_fn