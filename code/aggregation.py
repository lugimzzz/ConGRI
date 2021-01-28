import torch
from torch.autograd import Variable
import os
import linecache
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import pandas as pd
import argparse
import torch.nn as nn
from torch.nn import init
from PIL import Image
from models import DeModel, ResNetModel, VggModel, DenseModel, GoogleModel


# training configuration parameters
class Config():
    batch_size = {
        'train': 32,
        'test': 32
    }
    train_number_epochs = 30
    step_size = 10
    num_workers = 4
    if_shuffle = {
        'train': True,
        'test': False
    }

    lr = 0.005


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


def feature_reading(tf_info, target_info, set):
    features = []

    for gene_a_b in set:

        gene_a = gene_a_b[0]
        gene_b = gene_a_b[1]
        label = gene_a_b[2]
        feature = []

        if gene_a in tf_info['names'] and gene_b in target_info['names']:

            gene_a_index = np.argwhere(tf_info['names'] == gene_a)
            gene_b_index = np.argwhere(target_info['names'] == gene_b)

            for i in range(gene_a_index.shape[0]):
                for j in range(gene_b_index.shape[0]):
                    for direction in ['lateral', 'dorsal', 'ventral']:
                        if tf_info['directions'][gene_a_index[i][0]] == direction and \
                                target_info['directions'][gene_b_index[j][0]] == direction:
                            gene_a_feature = tf_info['embed'][gene_a_index[i][0]]
                            gene_b_feature = target_info['embed'][gene_b_index[j][0]]
                            feature.append(np.concatenate([gene_a_feature, gene_b_feature]))

            if 0 == len(feature):
                continue
            feature = np.array(feature)
            features.append([feature.mean(axis=0), label])

    return features


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
            dataloaders = DataLoader(gene_pair, batch_size=32,
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


# csv reading
def data_prepare():

    # loading gene pair name and the corresponding label
    dataset_raw = pd.read_csv("benchmark_dataset.csv", header=None)
    dataset = []
    for i in range(dataset_raw.shape[0]):
        dataset.append(list(dataset_raw.iloc[i][:]))

    # loading gene image corresponding image file name list
    gene_image_raw = pd.read_csv("gene_images.csv", header=None)
    gene_image = []
    for ii in range(gene_image_raw.shape[0]):
        gene_image.append(list(gene_image_raw.iloc[ii][:]))

    # loading target gene name
    target_raw = pd.read_csv("target.csv", header=None)
    target = []

    for i in range(target_raw.shape[0]):
        target.append(target_raw.iloc[i][0])

    # loading tf gene name
    tf_raw = pd.read_csv("tf.csv", header=None)
    tf = []

    for i in range(tf_raw.shape[0]):
        tf.append(tf_raw.iloc[i][0])

    print('csv loaded')

    return dataset, gene_image, target, tf


# main function
def main():

    num = 'google2'

    # load csv data, model
    dataset, gene_image, target, tf = data_prepare()
    tf, target = [], []
    for i in range(len(dataset)):

        gene_a_b = dataset[i]
        lateral, dorsal, ventral = gene_pair_to_image_group_three(gene_a_b, gene_image)

        if 0 == lateral + dorsal + ventral:

            all_miss.append([gene_a_b[0], gene_a_b[1], gene_a_b[2]])

        else:
            if not gene_a_b[0] in tf:
                tf.append(gene_a_b[0])
            if not gene_a_b[1] in target:
                target.append(gene_a_b[1])
    tf = np.array(tf)
    target = np.array(target)
    column = ['tf']
    tf_info = pd.DataFrame(columns=column, data=tf)
    tf_info.to_csv('tf_ind.csv')

    column = ['target']
    target_info = pd.DataFrame(columns=column, data=target)
    target_info.to_csv('target_ind.csv')

    # model
    model = GoogleModel(pretrained=True)
    # model = ResNetModel(pretrained=False)
    # model = VggModel(pretrained=False)
    model.load_state_dict(torch.load('{}.pkl'.format(num)))
    model = model.to(device)
    model.eval()
    print('model loaded')

    feature_map, name, direct = feature_reading_all(model, tf, gene_image)
    np.savez('./{}/tf.npz'.format(num), embed=np.array(feature_map), names=np.array(name), directions=np.array(direct))
    feature_map, name, direct = feature_reading_all(model, target, gene_image)
    np.savez('./{}/target.npz'.format(num), embed=np.array(feature_map), names=np.array(name), directions=np.array(direct))

    print('embeding loaded')    

    tf_info = np.load('./{}/tf.npz'.format(num))
    target_info = np.load('./{}/target.npz'.format(num))

    # divide the benchmark into five folds
    five_fold = []
    for i in range(5):
        five_fold.append(list(dataset[int(i * len(dataset) / 5):int((i + 1) * len(dataset) / 5)]))

    # (train + valid) : test = 4 : 1, train : valid = 9 : 1
    for fold in range(1, 2):

        print('Round ' + str(fold + 1) + ' starts now!')

        # train and valid
        train_all_set = []
        for j in range(5):
            if j == fold:
                continue
            train_all_set += five_fold[j][:]

        train_set = train_all_set[0:int(len(train_all_set) / 10 * 9)]
        valid_set = train_all_set[int(len(train_all_set) / 10 * 9):-1]
        test_set = five_fold[fold]

        train_features = feature_reading(tf_info, target_info, train_set)
        valid_features = feature_reading(tf_info, target_info, valid_set)
        test_features = feature_reading(tf_info, target_info, test_set)

        np.savez('./{}/train.npz'.format(num), embed=np.array(train_features))
        np.savez('./{}/valid.npz'.format(num), embed=np.array(valid_features))
        np.savez('./{}/test.npz'.format(num), embed=np.array(test_features))
        print('training data loaded')


if __name__ == '__main__':
    main()