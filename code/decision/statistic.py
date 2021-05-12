import numpy as np
import pandas as pd
from PIL import Image

root_path = '/data/lugim/data/flyexpress/data/pic_data/'


def gene_pair_to_image_group_three(gene_a_b, gene_image):

    lateral = 0
    dorsal = 0
    ventral = 0

    gene_a = gene_a_b[0]
    gene_b = gene_a_b[1]

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
                    if i == 0:
                        lateral += 1
                    if i == 1:
                        dorsal += 1
                    if i == 2:
                        ventral += 1

    return lateral, dorsal, ventral


# find the corresponding gene image file names with 3 angles(for one gene)
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

    return dataset, gene_image


def main():

    # load data
    tf = []
    target = []
    all_info = []
    all_miss = []

    all_lateral_zero = 0
    all_dorsal_zero = 0
    all_ventral_zero = 0
    all_lateral_one = 0
    all_dorsal_one = 0
    all_ventral_one = 0
    tf_target_zero = 0
    tf_target_one = 0

    dataset, gene_image = data_prepare()

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
    '''
    if gene_a_b[2] == 0:
                tf_target_zero += 1
                all_info.append([gene_a_b[0], gene_a_b[1], gene_a_b[2], lateral, dorsal, ventral,
                                 lateral + dorsal + ventral])
                all_lateral_zero += lateral
                all_dorsal_zero += dorsal
                all_ventral_zero += ventral
            else:
                tf_target_one += 1
                all_info.append([gene_a_b[0], gene_a_b[1], gene_a_b[2], lateral, dorsal, ventral,
                                 lateral + dorsal + ventral])
                all_lateral_one += lateral
                all_dorsal_one += dorsal
                all_ventral_one += ventral

    one = all_lateral_one + all_dorsal_one + all_ventral_one
    zero = all_lateral_zero + all_dorsal_zero + all_ventral_zero

    print('image one: {} {} {} {}'.format(all_lateral_one, all_dorsal_one, all_ventral_one, one))
    print('image zero: {} {} {} {}'.format(all_lateral_zero, all_dorsal_zero, all_ventral_zero, zero))
    print('image sum: {} {} {} {}'.format(all_lateral_one + all_lateral_zero, all_dorsal_one + all_dorsal_zero,
                                          all_ventral_one + all_ventral_zero, one + zero))
    print('tf-target: {} {} {}'.format(tf_target_zero, tf_target_one, tf_target_zero + tf_target_one))
    print('tf: {} target: {}'.format(len(tf), len(target)))
    print(len(all_miss))
    print(np.array(all_miss))
    '''

    tf = np.array(tf)
    target = np.array(target)
    column = ['tf']
    tf_info = pd.DataFrame(columns=column, data=tf)
    tf_info.to_csv('tf.csv')

    column = ['target']
    target_info = pd.DataFrame(columns=column, data=target)
    target_info.to_csv('target.csv')


if __name__ == '__main__':
    main()