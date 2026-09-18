import os
import re
import warnings

import matplotlib
matplotlib.use('Agg')            # headless backend: safe inside a Flask worker thread
import matplotlib.pyplot as plt

import cv2
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from Common import *

warnings.filterwarnings("ignore")

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model.t7 is expected next to this file (not relative to wherever the server was started)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.t7")
CLASS_NAMES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

_net = None


def _load_net():
    """Load the emotion network once and reuse it for every request."""
    global _net
    if _net is None:
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
        except Exception:
            # Newer PyTorch refuses to unpickle extra objects by default.
            # Only do this because model.t7 is your own trusted file.
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        net = VGG('VGG19')
        net.load_state_dict(checkpoint['net'])
        net.to(device)
        net.eval()
        _net = net
    return _net


def _natural_key(name):
    # frame2.jpg must come before frame10.jpg
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def getSizes(directory):
    for file in os.listdir(directory):
        f_img = os.path.join(directory, file)
        if os.path.isfile(f_img):
            img = Image.open(f_img).convert('RGB')
            img = cv2.resize(np.array(img), dsize=(227, 227), interpolation=cv2.INTER_CUBIC)
            img = Image.fromarray(img.astype(np.uint8))
            img.save(f_img)


def getEmotions(directory):
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    cut_size = 44
    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    files = [f for f in sorted(os.listdir(directory), key=_natural_key)
             if os.path.isfile(os.path.join(directory, f))]
    if not files:
        raise ValueError("No frames were extracted from the video.")

    net = _load_net()
    predictionList_num = []
    predictionList_name = []

    with torch.no_grad():
        for filename in files:
            raw_img = np.array(Image.open(os.path.join(directory, filename)).convert('RGB'))
            gray = rgb2gray(raw_img)
            gray = cv2.resize(gray, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
            img = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

            inputs = transform_test(img)                 # (ncrops, c, h, w)
            ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w).to(device)

            outputs = net(inputs)
            outputs_avg = outputs.view(ncrops, -1).mean(0)   # average over the crops
            predicted = int(torch.argmax(outputs_avg).item())

            predictionList_num.append(predicted)
            predictionList_name.append(CLASS_NAMES[predicted])

    return predictionList_num, predictionList_name


def getMax(emotion_count, emotion_list):
    sns.barplot(x=emotion_list, y=emotion_count, palette="husl")
    plt.xticks(rotation=15)


def timeTrends(emotion_list, emotions, duration, PLOTSDIR):
    xranges = np.linspace(0, duration, len(emotions))
    sns.scatterplot(x=xranges, y=emotions)
    plt.title("Emotion Detection (By Frame)")
    plt.yticks([0, 1, 2, 3, 4, 5, 6], emotion_list)
    plt.xlabel("Time (seconds)")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(PLOTSDIR, 'stage1_emotions.png'))
    plt.clf()


def getEmfromVideo(filepath, duration, PLOTSDIR):
    emotion_list = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotions, emotion_lists = getEmotions(filepath)
    timeTrends(emotion_list, emotions, duration, PLOTSDIR)
    return emotion_lists