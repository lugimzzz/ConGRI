import torch
from torch.autograd import Variable
import os
import linecache
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data import Dataset_Load
from loss import ContrastiveLoss
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import argparse
import datetime
import time
from utils import ModelSaver
from eval import evaluate, calculate_accuracy, test_evaluate, calculate_evaluation, evaluate_train
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.nn.modules.distance import PairwiseDistance
from models import ResNetModel, VggModel
from Dataset import LoadImagePair, LoadImageSet


# training configuration parameters
class Config():

    batch_size = {
        'train': 16,
        'valid': 16,
        'test': 16
    }
    train_number_epochs = 50
    step_size = 10
    num_workers = 4
    if_shuffle = {
        'train': True,
        'valid': False,
        'test': False
    }

    lr = 0.000005


# choose GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# l2 dist
l2_dist = PairwiseDistance(2)


# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# save the model weight with the highest accuracy
modelsaver = ModelSaver()


train_model = {
    '18': resnet18(pretrained=True),
    '34': resnet34(pretrained=True),
    '50': resnet50(pretrained=True),
    '101': resnet101(pretrained=True)
}


test_model = {
    '18': resnet18(pretrained=False),
    '34': resnet34(pretrained=False),
    '50': resnet50(pretrained=False),
    '101': resnet101(pretrained=False)
}


def save_if_best_accuracy(acc, state_dict, fold):
    modelsaver.save_if_best(acc, state_dict, fold)


# save the last epoch result
def save_last_checkpoint(state):
    torch.save(state, './result/last_checkpoint.pth')


# train and valid
def train_valid(train_valid_set, fold):

    acc = []
    best_th = []

    # load data
    dataloaders = {
        x: DataLoader(train_valid_set[x], batch_size=Config.batch_size[x], shuffle=Config.if_shuffle[x],
                      num_workers=Config.num_workers)
        for x in ['train', 'valid']}

    # model, loss, optimizer
    model = VggModel(pretrained=True)
    model = model.to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.lr)

    # adjust the learning rate every step_size
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.step_size, gamma=0.1)
    print('-'*20)
    print(Config.batch_size, Config.lr, Config.train_number_epochs, Config.step_size)
    print(model.model.classifier)

    # start to train and valid every epoch
    for epoch in range(0, Config.train_number_epochs):

        accuracy = {}
        best_threshold = {}
        roc_auc = 0.0
        avg_loss_contrastive = {}
        acc_train = 0.0

        for phase in ['train', 'valid']:

            all_dist = []
            loss_contrastive_sum = 0.0

            if phase == 'train':
                # scheduler.step()
                # if scheduler.last_epoch % scheduler.step_size == 0:
                #    print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
                model.train()
            else:
                model.eval()

            for i, data in enumerate(dataloaders[phase], 0):

                # data trained in GPU
                img0, img1, label = data
                img0, img1, label = Variable(img0).to(device), Variable(img1).to(device), Variable(label).to(device)

                # forward propagation
                output1, output2 = model(img0), model(img1)
                loss_contrastive = criterion(output1, output2, label)

                # L2 distance
                euclidean_distance = l2_dist.forward(output1, output2)
                dist_batch = euclidean_distance.data.cpu().numpy().flatten()
                label_batch = label.data.cpu().numpy().flatten()

                for ii in range(len(dist_batch)):
                    all_dist.append([dist_batch[ii], label_batch[ii]])

                if phase == 'train':

                    # loss and backward propagation
                    optimizer.zero_grad()
                    loss_contrastive.backward()
                    optimizer.step()

                loss_contrastive_sum += loss_contrastive.item()

            all_dist = np.array(all_dist)

            avg_loss_contrastive[phase] = loss_contrastive_sum * Config.batch_size[phase] / len(dataloaders[phase])

            if phase == 'valid':

                accuracy[phase], best_threshold[phase], roc_auc = evaluate(fold, epoch, all_dist)
                acc.append(accuracy[phase])
                best_th.append(best_threshold[phase])
                _, _, acc_train = calculate_accuracy(best_threshold['train'], all_dist)
                # to be done, save best accuracy according to training set threshold
                save_if_best_accuracy(accuracy[phase], model.state_dict(), fold)
                save_last_checkpoint({'epoch': epoch,
                                      'state_dict': model.state_dict(),
                                      'optimizer_state': optimizer.state_dict(),
                                      'accuracy': accuracy,
                                      'loss': avg_loss_contrastive['valid']
                                      })
            else:
                accuracy[phase], best_threshold[phase] = evaluate_train(all_dist)

            tpr_test, fpr_test, acc_test, f1_test, tp_test, fp_test, true_count, false_count = calculate_evaluation(
                best_threshold[phase], all_dist)

            # print loss, acc, ...
            print('accuracy: {:.4f} tpr: {:.4f} fpr: {:.4f} f1: {:.4f}  tp: {:.4f} fp: {:.4f}'.format(acc_test,
                                                                                                      tpr_test,
                                                                                                      fpr_test, f1_test,
                                                                                                      tp_test, fp_test))

            print('true distribution: ', end='')
            for i in range(len(true_count)):
                print('{:.2f} '.format(true_count[i]), end='')
            print('')
            print('false distribution: ', end='')
            for i in range(len(false_count)):
                print('{:.2f} '.format(false_count[i]), end='')
            print('\n')

        # print loss, acc, ...
        print('{}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f} {:.3f}'.format(epoch,
                                                                                         avg_loss_contrastive['train'],
                                                                                         accuracy['train'],
                                                                                         best_threshold['train'],
                                                                                         acc_train,
                                                                                         avg_loss_contrastive['valid'],
                                                                                         accuracy['valid'],
                                                                                         best_threshold['valid'],
                                                                                         roc_auc))
        print('\n')
        # save the model weight parameters
        if epoch % 5 == 0:
            torch.save(model.state_dict(), './result/{}_{}_net.pkl'.format(fold, epoch))

        # model
        column = ['dist', 'label']
        valid = pd.DataFrame(columns=column, data=all_dist)
        valid.to_csv('./result/{}_{}_dist.csv'.format(fold, epoch))

    torch.save(model.state_dict(), './result/{}_{}_net.pkl'.format(fold, Config.train_number_epochs))

    acc = np.array(acc)
    best_th = np.array(best_th)
    print('fold {}  best accuracy is epoch {} : {:.4f} . The best threshold is {:.4f}'.format(fold, np.argmax(acc),
                                                                                              acc[np.argmax(acc)],
                                                                                              best_th[np.argmax(acc)]))
    return best_th[np.argmax(acc)]


def test(dataset, th, gene_image, fold, phase):

    # model
    model = VggModel(pretrained=False)
    model.load_state_dict(torch.load('{}_best_state.pkl'.format(fold)))
    model = model.to(device)
    model.eval()

    # loss
    loss = ContrastiveLoss()

    # initialization
    loss_contrastive_sum = 0.0
    dist_all= []
    dist_deci = []
    dist_avg = []
    # read the gene pair one by one
    for gene_a_b in dataset:

        dist_pair = []
        # change gene pair name into gene pair file name list

        gene_pair = LoadImagePair(transform, gene_a_b, gene_image)

        dataloaders = DataLoader(gene_pair, batch_size=Config.batch_size['test'], shuffle=Config.if_shuffle['test'],
                   num_workers=Config.num_workers)

        for i, data in enumerate(dataloaders, 0):

            # data trained in GPU
            img0, img1, label = data
            img0, img1, label = Variable(img0).to(device), Variable(img1).to(device), Variable(label).to(device)

            # forward propagation
            output1, output2 = model(img0), model(img1)

            loss_contrastive = loss(output1, output2, label)
            loss_contrastive_sum += loss_contrastive.item()

            distance = l2_dist.forward(output1, output2)
            dist_batch = distance.data.cpu().numpy().flatten()
            label_batch = label.data.cpu().numpy().flatten()

            for ii in range(len(dist_batch)):
                dist_all.append([dist_batch[ii], label_batch[ii]])
                dist_pair.append([dist_batch[ii], label_batch[ii]])

        dist_pair = np.array(dist_pair)
        if dist_pair.shape[0] == 0:
            continue

        dist_avg.append([np.mean(dist_pair[:, 0]), dist_pair[0, 1]])
        dist_deci.append(test_evaluate(th, dist_pair))

    avg_loss_contrastive = loss_contrastive_sum/len(dataset)
    print('fold: {}   {}   loss: {:.4f}'.format(fold, phase, avg_loss_contrastive))
    dist_all = np.array(dist_all)
    dist_avg = np.array(dist_avg)
    dist_deci = np.array(dist_deci)

    dist = {

        'image pair': dist_all,
        'gene pair average': dist_avg,
        'gene pair co-decision': dist_deci

    }

    for type in ('image pair', 'gene pair average', 'gene pair co-decision'):
        dist_t = dist[type]
        tpr_test, fpr_test, acc_test, f1_test, tp_test, fp_test, true_count, false_count = calculate_evaluation(th,
                                                                                                                dist_t)

        # print loss, acc, ...
        print('{}  accuracy: {:.4f} tpr: {:.4f} fpr: {:.4f} f1: {:.4f}  tp: {:.4f} fp: {:.4f}'.format(type, acc_test,
                                                                                                      tpr_test,
                                                                                                      fpr_test, f1_test,
                                                                                                      tp_test, fp_test))

        print('true distribution: ', end='')
        for i in range(len(true_count)):
            print('{:.4f} '.format(true_count[i]), end='')
        print('')
        print('false distribution: ', end='')
        for i in range(len(false_count)):
            print('{:.4f} '.format(false_count[i]), end='')
        print('')

    column = ['dist', 'label']
    test_output = pd.DataFrame(columns=column, data=dist_all)
    test_output.to_csv('./result/{}_{}_all_dist.csv'.format(fold, phase))


def data_prepare():

    # loading gene pair name and the corresponding label
    dataset_raw = pd.read_csv("benchmark_dataset.csv", header=None)
    # dataset_raw = pd.read_csv("sub_benchmark.csv", header=None)
    dataset = []
    for i in range(dataset_raw.shape[0]):
        dataset.append(list(dataset_raw.iloc[i][:]))

    # loading gene image corresponding image file name list
    gene_image_raw = pd.read_csv("gene_images.csv",header=None)
    gene_image = []
    for ii in range(gene_image_raw.shape[0]):
        gene_image.append(list(gene_image_raw.iloc[ii][:]))

    # loading independent test set name and the corresponding label
    ind_test_set_raw = pd.read_csv("independent_test_set.csv", header=None)
    ind_test_set = []
    for i in range(ind_test_set_raw.shape[0]):
        ind_test_set.append(list(ind_test_set_raw.iloc[i][:]))

    return dataset, gene_image, ind_test_set


# main function
def main():

    # load data
    dataset, gene_image, ind_test_set = data_prepare()

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

        train_valid_set = {
            'train': LoadImageSet(transform, train_set, gene_image),
            'valid': LoadImageSet(transform, valid_set, gene_image)
        }

        th = train_valid(train_valid_set, fold)
        # th = 0.5
        # test
        test_set = five_fold[fold]

        # benchmark test data set
        test(test_set, th, gene_image, fold, 'benchmark_test')

        # independent test data set
        test(ind_test_set, th, gene_image, fold, 'independent_test')

        print(f'{fold+1} : done')

    print('done')


if __name__ == '__main__':
    main()