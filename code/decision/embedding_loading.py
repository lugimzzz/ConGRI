import torch
from torch.autograd import Variable
import os
import linecache
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loss import ContrastiveLoss
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import argparse
import torch.nn as nn
import torch.utils.data as Data
from utils import ModelSaver
from eval import evaluate, calculate_accuracy, test_evaluate, calculate_evaluation, evaluate_train
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.nn.modules.distance import PairwiseDistance
from models import ResNetModel, VggModel, DeModel
from PIL import Image

root_path = '/data/lugim/data/flyexpress/data/pic_data/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# choose GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class LoadImagePairThree(Dataset):

    def __init__(self, transform=None, gene=[]):

        self.transform = transform
        self.gene = gene
        self.gene_num = len(self.gene)

    def __getitem__(self, index):

        gene_img = Image.open(root_path + self.gene[index])

        if self.transform:

            gene_img = self.transform(gene_img)

        return gene_img

    def __len__(self):
        return self.gene_num


def gene_to_image_group_three(gene, gene_image):

    image_gene_pair_lateral = []
    image_gene_pair_dorsal = []
    image_gene_pair_ventral = []

    gene_batch = find_image_group(gene_image, gene)

    for i in range(3):
        for j in range(len(gene_batch[i])):
            try:
                Image.open(root_path + gene_batch[i][j])

            except Exception:
                continue

            else:
                if i == 0:
                    image_gene_pair_lateral.append(gene_batch[i][j])
                if i == 1:
                    image_gene_pair_dorsal.append(gene_batch[i][j])
                if i == 2:
                    image_gene_pair_ventral.append(gene_batch[i][j])

    return image_gene_pair_lateral, image_gene_pair_dorsal, image_gene_pair_ventral


def find_image_group(gene_images, gene):

    image_lateral = []
    image_dorsal = []
    image_ventral = []

    for row in gene_images:
        if row[6] == 6 and (row[0] == gene or row[1] == gene or row[2] == gene or row[3] == gene or row[4] == gene):

            if row[7] == 'lateral':
                name = row[5]
                img_name = str(name.split('/')[-1].split('.')[0].split('u')[-1])+'_s.bmp'
                image_lateral.append(img_name)

            if row[7] == 'dorsal':
                name = row[5]
                img_name = str(name.split('/')[-1].split('.')[0].split('u')[-1]) + '_s.bmp'
                image_dorsal.append(img_name)

            if row[7] == 'ventral':
                name = row[5]
                img_name = str(name.split('/')[-1].split('.')[0].split('u')[-1])+'_s.bmp'
                image_ventral.append(img_name)

    return [image_lateral, image_dorsal, image_ventral]


# csv reading
def data_prepare():

    # loading gene image corresponding image file name list
    gene_image_raw = pd.read_csv("gene_images.csv", header=None)
    gene_image = []
    for ii in range(gene_image_raw.shape[0]):
        gene_image.append(list(gene_image_raw.iloc[ii][:]))

    # loading tf gene name
    # tf_raw = pd.read_csv("tf1.csv", header=None)
    tf_raw = pd.read_csv("tf.csv", header=None)
    tf = []
    for i in range(tf_raw.shape[0]):
        tf.append(tf_raw.iloc[i][0])

    # loading target gene name
    # target_raw = pd.read_csv("tf1.csv", header=None)
    target_raw = pd.read_csv("target.csv", header=None)
    target = []

    for i in range(target_raw.shape[0]):
        target.append(target_raw.iloc[i][0])

    print('csv loaded')
    return gene_image, tf, target


def feature_reading(trained_models, set, gene_image):

    feature_map = []
    name = []
    direct = []

    for gene in set:

        image_gene_pair_lateral, image_gene_pair_dorsal, image_gene_pair_ventral = gene_to_image_group_three(
            gene, gene_image)

        if 0 == len(image_gene_pair_lateral) + len(image_gene_pair_dorsal) + len(image_gene_pair_ventral):
            print('we can not find the corresponding image for :', gene)
            continue

        image_gene_pair = {
            'lateral': image_gene_pair_lateral,
            'dorsal': image_gene_pair_dorsal,
            'ventral': image_gene_pair_ventral
        }
        directions = ['lateral', 'dorsal', 'ventral']
        for direction in directions:

            if 0 == len(image_gene_pair[direction]):
                continue

            gene_pair = LoadImagePairThree(transform, image_gene_pair[direction])
            dataloaders = DataLoader(gene_pair, batch_size=16,
                                     shuffle=False, num_workers=1)

            for i, img in enumerate(dataloaders, 0):

                img = img.to(device)
                model = trained_models[direction]
                output = model(img)
                feature = output.data.cpu().numpy()
                for ii in range(len(feature)):

                    feature_map.append(feature[ii])
                    name.append(gene)
                    direct.append(direction)

    return feature_map, name, direct


def feature_reading_all(model, set, gene_image):

    feature_map = []
    name = []
    direct = []

    for gene in set:

        image_gene_pair_lateral, image_gene_pair_dorsal, image_gene_pair_ventral = gene_to_image_group_three(
            gene, gene_image)

        if 0 == len(image_gene_pair_lateral) + len(image_gene_pair_dorsal) + len(image_gene_pair_ventral):
            print('we can not find the corresponding image for :', gene)
            continue

        image_gene_pair = {
            'lateral': image_gene_pair_lateral,
            'dorsal': image_gene_pair_dorsal,
            'ventral': image_gene_pair_ventral
        }
        directions = ['lateral', 'dorsal', 'ventral']
        for direction in directions:

            if 0 == len(image_gene_pair[direction]):
                continue

            gene_pair = LoadImagePairThree(transform, image_gene_pair[direction])
            dataloaders = DataLoader(gene_pair, batch_size=16,
                                     shuffle=False, num_workers=1)

            for i, img in enumerate(dataloaders, 0):

                img = img.to(device)
                output = model(img)
                feature = output.data.cpu().numpy()
                for ii in range(len(feature)):

                    feature_map.append(feature[ii])
                    name.append(gene)
                    direct.append(direction)

    return feature_map, name, direct


def first_level_model():
    # model trained by different direction image
    model_lateral = VggModel(pretrained=False)
    model_lateral.load_state_dict(torch.load('lateral.pkl'))
    model_lateral = model_lateral.to(device)
    model_lateral.eval()

    model_ventral = VggModel(pretrained=False)
    model_ventral.load_state_dict(torch.load('ventral.pkl'))
    model_ventral = model_ventral.to(device)
    model_ventral.eval()

    model_dorsal = VggModel(pretrained=False)
    model_dorsal.load_state_dict(torch.load('dorsal.pkl'))
    model_dorsal = model_dorsal.to(device)
    model_dorsal.eval()

    trained_models = {
        'lateral': model_lateral,
        'dorsal': model_dorsal,
        'ventral': model_ventral
    }

    print('Pretrained model loading finished')

    return trained_models


# main function
def main():

    # load csv data, model
    gene_image, tf, target = data_prepare()

    model = VggModel(pretrained=False)
    model.load_state_dict(torch.load('all.pkl'))
    model = model.to(device)
    model.eval()

    feature_map, name, direct = feature_reading_all(model, tf, gene_image)
    np.savez('tf_all.npz', embed=np.array(feature_map), names=np.array(name), directions=np.array(direct))
    feature_map, name, direct = feature_reading_all(model, target, gene_image)
    np.savez('target_all.npz', embed=np.array(feature_map), names=np.array(name), directions=np.array(direct))

    print('training data loaded')


'''
    trained_models = first_level_model()

    feature_map, name, direct = feature_reading(trained_models, tf, gene_image)
    np.savez('tf.npz', embed=np.array(feature_map), names=np.array(name), directions=np.array(direct))

    feature_map, name, direct = feature_reading(trained_models, target, gene_image)
    np.savez('target.npz', embed=np.array(feature_map), names=np.array(name), directions=np.array(direct))    
'''


if __name__ == '__main__':
    main()