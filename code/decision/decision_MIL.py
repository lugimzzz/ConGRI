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
from torch.nn import init

# training configuration parameters
class Config():
    batch_size = {
        'train': 16,
        'test': 16
    }
    train_number_epochs = 30
    step_size = 10
    num_workers = 4
    if_shuffle = {
        'train': True,
        'test': False
    }

    lr = 0.01


# define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.05)
        # m.weight.data.uniform_(0.0, 1.0)
        # m.bias.data.fill_(0)
        # n = m.in_features
        # y = 1.0 / np.sqrt(n)
        # m.weight.data.uniform_(-y, y)
        # m.bias.data.fill_(0)


# choose GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# save the model weight with the highest accuracy
modelsaver = ModelSaver()


def save_if_best_accuracy(acc, state_dict, fold):
    modelsaver.save_if_best(acc, state_dict, fold)


# save the latest epoch result
def save_last_checkpoint(state):
    torch.save(state, './result/last_checkpoint.pth')


# train and valid
def second_model(train_features, valid_features, test_features, fold):
    # model loss function optimizer
    model = DeModel()
    model = model.to(device)
    model.apply(weigth_init)
    cross_entropy_loss = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.lr)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    train_set = LoadFeatureSet(train_features)
    valid_set = LoadFeatureSet(valid_features)
    test_set = LoadFeatureSet(test_features)
    print(Config.batch_size, Config.lr, Config.train_number_epochs, Config.step_size)
    print(model.model)

    # data loading
    dataloaders = {
        'train': DataLoader(train_set, batch_size=Config.batch_size['train'], shuffle=Config.if_shuffle['train'],
                            num_workers=Config.num_workers),
        'valid': DataLoader(valid_set, batch_size=Config.batch_size['train'], shuffle=Config.if_shuffle['train'],
                            num_workers=Config.num_workers),
        'test': DataLoader(test_set, batch_size=Config.batch_size['train'], shuffle=Config.if_shuffle['train'],
                            num_workers=Config.num_workers)
    }

    # train
    for epoch in range(0, Config.train_number_epochs):

        avg_loss = {}
        loss_sum = 0.0
        print(epoch)
        for phase in ['train', 'valid', 'test']:

            all_de = []
            acc_train = 0.0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for i, data in enumerate(dataloaders[phase], 0):
                feature_input, label = data
                feature_input, label = feature_input.to(device), label.to(device)
                output = model(feature_input)
                output = output.squeeze(-1)
                loss_cross = cross_entropy_loss(output, label)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss_cross.backward()
                    optimizer.step()
                    # scheduler.step()

                loss_sum += loss_cross.item()
                output = output.data.cpu().numpy()
                label = label.data.cpu().numpy()
                for ii in range(output.shape[0]):
                    all_de.append([output[ii], label[ii]])

            avg_loss[phase] = loss_sum / len(dataloaders[phase])
            all_de = np.array(all_de)
            threshold = 0.6
            tp = np.sum((all_de[:, 0] > threshold) & (all_de[:, 1] == 1))
            fp = np.sum((all_de[:, 0] > threshold) & (all_de[:, 1] == 0))
            fn = np.sum((all_de[:, 0] <= threshold) & (all_de[:, 1] == 1))
            tn = np.sum((all_de[:, 0] <= threshold) & (all_de[:, 1] == 0))

            acc = float(tp + tn) / float(tp + fp + fn + tn)
            tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
            fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
            precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
            recall = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
            f1 = 0 if (precision + recall == 0) else 2 * float(recall * precision) / float(recall + precision)
            print('{} loss: {:.4f} acc: {:.4f} tpr: {:.4f} fpr: {:.4f} f1: {:.4f}'.format(phase, avg_loss[phase], acc, tpr, fpr, f1))
            if phase == 'test':
                save_if_best_accuracy(acc, model.state_dict(), fold)


# test
def second_model_test(test_features, fold):
    # model loss function optimizer
    model = DeModel()
    model = model.to(device)
    # model.load_state_dict(torch.load('./result/{}_best_state.pkl'.format(fold)))
    model.load_state_dict(torch.load('./result/7509.pkl'))
    model.eval()
    cross_entropy_loss = nn.BCELoss()
    test_set = LoadFeatureSet(test_features)

    # data loading
    dataloaders = DataLoader(test_set, batch_size=Config.batch_size['test'], shuffle=Config.if_shuffle['test'],
                             num_workers=Config.num_workers)
    loss_sum = 0.0
    all_de = []
    # test
    for i, data in enumerate(dataloaders, 0):
        feature_input, label = data
        feature_input, label = feature_input.to(device), label.to(device)
        output = model(feature_input)
        output = output.squeeze(-1)
        loss_cross = cross_entropy_loss(output, label)

        loss_sum += loss_cross.item()
        output = output.data.cpu().numpy()
        label = label.data.cpu().numpy()
        for ii in range(output.shape[0]):
            all_de.append([output[ii], label[ii]])

    avg_loss = loss_sum / len(dataloaders)
    all_de = np.array(all_de)
    len_dist = len(all_de)
    for th in [0.5]:

        threshold = th
        tp = np.sum((all_de[:, 0] > threshold) & (all_de[:, 1] == 1))
        fp = np.sum((all_de[:, 0] > threshold) & (all_de[:, 1] == 0))
        fn = np.sum((all_de[:, 0] <= threshold) & (all_de[:, 1] == 1))
        tn = np.sum((all_de[:, 0] <= threshold) & (all_de[:, 1] == 0))

        acc = float(tp + tn) / float(tp + fp + fn + tn)
        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
        recall = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        f1 = 0 if (precision + recall == 0) else 2 * float(recall * precision) / float(recall + precision)

        true_eucli_dist = all_de[((all_de[:, 0] > threshold) & (all_de[:, 1] == 1)) | (
                (all_de[:, 0] <= threshold) & (all_de[:, 1] == 0))]
        false_eucli_dist = all_de[((all_de[:, 0] > threshold) & (all_de[:, 1] == 0)) | (
                (all_de[:, 0] <= threshold) & (all_de[:, 1] == 1))]
        true_count = []
        false_count = []
        step = 0.2
        for i in np.arange(0, 1, step):
            true_count.append(np.sum(true_eucli_dist[:, 0] < (i + step)) - np.sum(true_eucli_dist[:, 0] < i))
            false_count.append(np.sum(false_eucli_dist[:, 0] < (i + step)) - np.sum(false_eucli_dist[:, 0] < i))

        true_count = np.array(true_count) / len_dist
        false_count = np.array(false_count) / len_dist

        print('test  loss: {:.4f} acc: {:.4f} recall: {:.4f} pre: {:.4f} f1: {:.4f}'.format(avg_loss, acc, recall, precision, f1))
        print(tp,fp,tn,fn)


# csv reading
def data_prepare():
    # loading gene pair name and the corresponding label
    dataset_raw = pd.read_csv("benchmark_dataset.csv", header=None)
    # dataset_raw = pd.read_csv("sub_benchmark.csv", header=None)
    dataset = []
    for i in range(dataset_raw.shape[0]):
        dataset.append(list(dataset_raw.iloc[i][:]))

    return dataset


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
            features.append([feature.max(axis=0), label])

    # features = np.array(features)

    return features


class LoadFeatureSet(Dataset):

    def __init__(self, features=[]):
        self.features = features[:, 0]
        self.labels = features[:, 1]
        self.feature_len = len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        feature = torch.from_numpy(feature).float()
        label = torch.tensor(label).float()

        return feature, label

    def __len__(self):
        return self.feature_len


# main function
def main():
    # load csv data, model
    train_features = np.load('train_mean.npz', allow_pickle=True)
    valid_features = np.load('valid_mean.npz', allow_pickle=True)
    test_features = np.load('test_mean.npz', allow_pickle=True)
    print('data loaded')
    print(len(train_features['embed']), len(valid_features['embed']), len(test_features['embed']))
    # train
    # second_model(train_features['embed'], valid_features['embed'], test_features['embed'], 1)

    # test
    second_model_test(test_features['embed'], 1)

    print('done')
    '''
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
        print('data loaded')

        np.savez('train.npz', embed=np.array(train_features))
        np.savez('valid.npz', embed=np.array(valid_features))
        np.savez('test.npz', embed=np.array(test_features))

        # second model training
        # second_model(train_features, valid_features, fold)

        # test
        # test_set = five_fold[fold]
        # test_features = feature_reading(tf_info, target_info, test_set)
        print(f'{fold + 1} : done')

    print('done')
    '''


if __name__ == '__main__':
    main()
