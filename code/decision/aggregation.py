import numpy as np
import pandas as pd


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
            features.append([feature.mean(axis=0), label])

    return features


# main function
def main():
    # load csv data, model
    dataset = data_prepare()
    tf_info = np.load('tf_all.npz')
    target_info = np.load('target_all.npz')

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

        np.savez('train_mean_all.npz', embed=np.array(train_features))
        np.savez('valid_mean_all.npz', embed=np.array(valid_features))
        np.savez('test_mean_all.npz', embed=np.array(test_features))

        print(f'{fold + 1} : done')

    print('done')


if __name__ == '__main__':
    main()
