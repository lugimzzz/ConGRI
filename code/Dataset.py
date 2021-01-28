from PIL import Image
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
root_path = '/data/lugim/data/flyexpress/data/pic_data/'


# find the corresponding gene image file names with 3 angles(for one gene)
def find_image_group(gene_images, gene):

    image_lateral = []
    image_dorsal = []
    image_ventral = []

    for row in gene_images:
        if row[6] == 6 and (row[0] == gene or row[1] == gene or row[2] == gene or row[3] == gene or row[4] == gene):
            if row[7] == 'dorsal':
                name = row[5]
                img_name = str(name.split('/')[-1].split('.')[0].split('u')[-1]) + '_s.bmp'
                image_dorsal.append(img_name)

            if row[7] == 'lateral':
                name = row[5]
                img_name = str(name.split('/')[-1].split('.')[0].split('u')[-1])+'_s.bmp'
                image_lateral.append(img_name)

            if row[7] == 'ventral':
                name = row[5]
                img_name = str(name.split('/')[-1].split('.')[0].split('u')[-1])+'_s.bmp'
                image_ventral.append(img_name)

    return [image_lateral, image_dorsal, image_ventral]


def gene_pair_set_to_image_group(gene_a_b_set, gene_image):

    image_gene_pair = []
    for ii in range(len(gene_a_b_set)):

        gene_a_b = gene_a_b_set[ii]
        gene_a = gene_a_b[0]
        gene_b = gene_a_b[1]
        label = gene_a_b[2]

        gene_a_batch = find_image_group(gene_image, gene_a)
        gene_b_batch = find_image_group(gene_image, gene_b)

        for i in range(3):
            for j in range(len(gene_a_batch[i])):
                for k in range(len(gene_b_batch[i])):

                    try:
                        Image.open(root_path + gene_a_batch[i][j])
                        Image.open(root_path + gene_b_batch[i][k])
                    except Exception:
                        continue

                    else:
                        image_gene_pair.append([gene_a_batch[i][j], gene_b_batch[i][k], label])

    return image_gene_pair


class LoadImageSet(Dataset):

    def __init__(self, transform=None, gene_a_b_set = [], gene_image = []):

        self.transform = transform
        self.pairs = gene_pair_set_to_image_group(gene_a_b_set, gene_image)
        self.pairs_num = len(self.pairs)

    def __getitem__(self, index):

        gene_1 = self.pairs[index][0]
        gene_1_img = Image.open(root_path + gene_1)

        gene_2 = self.pairs[index][1]
        gene_2_img = Image.open(root_path + gene_2)

        label = self.pairs[index][2]
        #print('0', gene_1_img)
        #print('1', np.array(gene_1_img))
        if self.transform:

            gene_1_img = self.transform(gene_1_img)
            gene_2_img = self.transform(gene_2_img)

        #print('2', gene_1_img)
        #print('3', self.transform(np.array(gene_1_img)))
        #print('4', self.transform(np.array(gene_1_img)/255))
        return gene_1_img, gene_2_img, label

    def __len__(self):
        return self.pairs_num


def gene_pair_to_image_group(gene_a_b, gene_image):

    image_gene_pair = []
    gene_a = gene_a_b[0]
    gene_b = gene_a_b[1]
    label = gene_a_b[2]

    gene_a_batch = find_image_group(gene_image, gene_a)
    gene_b_batch = find_image_group(gene_image, gene_b)

    for i in range(3):
        for j in range(len(gene_a_batch[i])):
            for k in range(len(gene_b_batch[i])):

                try:
                    Image.open(root_path + gene_a_batch[i][j])
                    Image.open(root_path + gene_b_batch[i][k])
                except Exception:
                    continue

                else:
                    image_gene_pair.append([gene_a_batch[i][j], gene_b_batch[i][k], label])

    return image_gene_pair


class LoadImagePair(Dataset):

    def __init__(self, transform=None, gene_a_b = [], gene_image = []):

        self.transform = transform
        self.pairs = gene_pair_to_image_group(gene_a_b, gene_image)
        self.pairs_num = len(self.pairs)

    def __getitem__(self, index):

        gene_1 = self.pairs[index][0]
        gene_1_img = Image.open(root_path + gene_1)

        gene_2 = self.pairs[index][1]
        gene_2_img = Image.open(root_path + gene_2)

        label = self.pairs[index][2]

        if self.transform:

            gene_1_img = self.transform(gene_1_img)
            gene_2_img = self.transform(gene_2_img)

        return gene_1_img, gene_2_img, label

    def __len__(self):
        return self.pairs_num
