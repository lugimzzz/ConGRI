import numpy as np
import pandas as pd
import os
import csv
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
from Dataset import test_Load
from loss import ContrastiveLoss
from data import Dataset_Load
from torch.nn.modules.distance import PairwiseDistance
from eval import evaluate, calculate_accuracy, test_evaluate
import torch
from torch.autograd import Variable
from torchvision.models import resnet50
from torchvision import transforms
from model import ResNet50
dist = PairwiseDistance(2)
class Config():

    batch_size = 1
    train_number_epochs = 30
    step_size = 10
    num_workers = 4
    if_shuffle = False

# choose GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform
transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def test(dataset, th):

    # loading gene image corresponding image file name list
    gene_image_raw = pd.read_csv("gene_images.csv",header=None)
    gene_image = []
    for ii in range(gene_image_raw.shape[0]):
        gene_image.append(list(gene_image_raw.iloc[ii][:]))

    # model
    model = ResNet50(pretrained=False)
    model.load_state_dict(torch.load('best_state.pkl'))
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

        gene_pair = test_Load(transform_img, gene_a_b, gene_image)

        dataloaders = DataLoader(gene_pair, batch_size=Config.batch_size, shuffle=Config.if_shuffle,
                   num_workers=Config.num_workers)
        for i, data in enumerate(dataloaders, 0):

            # data trained in GPU
            img0, img1, label = data
            img0, img1, label = Variable(img0).to(device), Variable(img1).to(device), Variable(label).to(device)

            # forward propagation
            output1, output2 = model(img0), model(img1)

            loss_contrastive = loss(output1, output2, label)
            loss_contrastive_sum += loss_contrastive.item()

            distance = dist.forward(output1, output2)
            dist_all.append([distance.item(), label.item()])
            dist_pair.append([distance.item(), label.item()])

        dist_pair = np.array(dist_pair)
        if dist_pair.shape[0] == 0:
            continue

        dist_avg.append([np.mean(dist_pair[:, 0]), dist_pair[0, 1]])
        dist_deci.append(test_evaluate(th, dist_pair))


    avg_loss_contrastive = loss_contrastive_sum/ len(dataset)
    dist_all = np.array(dist_all)
    dist_avg = np.array(dist_avg)
    dist_deci = np.array(dist_deci)
    tpr_test, fpr_test, acc_test = calculate_accuracy(th, dist_all)
    tpr_test_avg, fpr_test_avg, acc_test_avg = calculate_accuracy(th, dist_avg)
    tpr_test_deci, fpr_test_deci, acc_test_deci = calculate_accuracy(th, dist_deci)

    # print loss, acc, ...
    print(f' loss: {avg_loss_contrastive}')
    print(f' image pair evaluation   accuracy: {acc_test} tpr: {tpr_test} fpr: {fpr_test}')
    print(f' gene pair evaluation with average distance   accuracy: {acc_test_avg} tpr: {tpr_test_avg} fpr: {fpr_test_avg}')
    print(f' gene pair evaluation with decision   accuracy: {acc_test_deci} tpr: {tpr_test_deci} fpr: {fpr_test_deci}')

    column = ['dist', 'label']
    test = pd.DataFrame(columns=column, data=dist_all)
    test.to_csv('./result/all_dist.csv')



if __name__ == '__main__':

    dataset_raw = pd.read_csv("benchmark_dataset.csv", header=None)
    dataset = []

    for i in range(dataset_raw.shape[0]):
        dataset.append(list(dataset_raw.iloc[i][:]))
    test_5 = dataset[int(4*len(dataset)/5): -1]
    test(test_5, 0.73)

    dataset_raw = pd.read_csv("independent_test_set.csv", header=None)
    dataset = []
    for i in range(dataset_raw.shape[0]):
        dataset.append(list(dataset_raw.iloc[i][:]))
    test(dataset, 0.73)