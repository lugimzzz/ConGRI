import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def evaluate(fold, epoch, eucli_dist):

    # Calculate evaluation metrics
    thresholds = np.arange(0, 1, 0.001)
    accuracy, best_threshold, roc_auc, tprs, fprs= calculate_roc(thresholds, eucli_dist)
    plot_roc(fprs, tprs, figure_name='./result/roc_valid_epoch_{}_{}.png'.format(fold,epoch))
    return accuracy, best_threshold, roc_auc


def evaluate_train(eucli_dist):

    # Calculate evaluation metrics
    thresholds = np.arange(0, 1, 0.001)
    accuracy, best_threshold, _, _, _ = calculate_roc(thresholds, eucli_dist)

    return accuracy, best_threshold


def calculate_roc(thresholds, eucli_dist):

    nrof_thresholds = len(thresholds)
    tprs = np.zeros(nrof_thresholds)
    fprs = np.zeros(nrof_thresholds)
    accs = np.zeros(nrof_thresholds)
    #f1s = np.zeros(nrof_thresholds)

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], accs[threshold_idx]= calculate_accuracy(threshold, eucli_dist)

    roc_auc = auc(fprs, tprs)
    # distances = np.sqrt(fprs**2+(tprs-1)**2)
    # best_threshold_index = np.argmin(distances)
    best_threshold_index = np.argmax(accs)
    _, _, accuracy = calculate_accuracy(thresholds[best_threshold_index], eucli_dist)

    return  accuracy, thresholds[best_threshold_index],  roc_auc, tprs, fprs


def calculate_evaluation(threshold, eucli_dist):

    tp = np.sum((eucli_dist[:, 0] <= threshold) & (eucli_dist[:, 1] == 1))
    fp = np.sum((eucli_dist[:, 0] <= threshold) & (eucli_dist[:, 1] == 0))
    fn = np.sum((eucli_dist[:, 0] > threshold) & (eucli_dist[:, 1] == 1))
    tn = np.sum((eucli_dist[:, 0] > threshold) & (eucli_dist[:, 1] == 0))

    acc = float(tp + tn) / len(eucli_dist)
    len_dist = len(eucli_dist)

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
    recall = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    f1 = 0 if (precision + recall == 0) else 2 * float(recall * precision) / float(recall + precision)

    tp_t = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
    fp_t = 0 if (tp + fp == 0) else float(fp) / float(tp + fp)

    true_eucli_dist = eucli_dist[((eucli_dist[:, 0] <= threshold) & (eucli_dist[:, 1] == 1)) | (
            (eucli_dist[:, 0] > threshold) & (eucli_dist[:, 1] == 0))]
    false_eucli_dist = eucli_dist[((eucli_dist[:, 0] <= threshold) & (eucli_dist[:, 1] == 0)) | (
            (eucli_dist[:, 0] > threshold) & (eucli_dist[:, 1] == 1))]
    true_count = []
    false_count = []
    step = 0.2
    for i in np.arange(0, 1, step):
        true_count.append(np.sum(true_eucli_dist[:, 0] < (i + step)) - np.sum(true_eucli_dist[:, 0] < i))
        false_count.append(np.sum(false_eucli_dist[:, 0] < (i + step)) - np.sum(false_eucli_dist[:, 0] < i))

    true_count.append(len(true_eucli_dist) - sum(true_count))
    false_count.append(len(false_eucli_dist) - sum(false_count))
    true_count = np.array(true_count) / len_dist
    false_count = np.array(false_count) / len_dist

    return tpr, fpr, acc, f1, tp_t, fp_t, true_count, false_count


def calculate_accuracy(threshold, eucli_dist):

    tp = np.sum((eucli_dist[:, 0] <= threshold) & (eucli_dist[:, 1] == 1))
    fp = np.sum((eucli_dist[:, 0] <= threshold) & (eucli_dist[:, 1] == 0))
    fn = np.sum((eucli_dist[:, 0] > threshold) & (eucli_dist[:, 1] == 1))
    tn = np.sum((eucli_dist[:, 0] > threshold) & (eucli_dist[:, 1] == 0))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / len(eucli_dist)

    return tpr, fpr, acc


def test_evaluate(threshold, eucli_dist):

    p = np.sum(eucli_dist[:, 0] <= threshold)
    n = np.sum(eucli_dist[:, 0] > threshold)
    rate = p / (p + n)
    if  rate > 0.4:
        pred = 0
    else:
        pred = 100
    return [pred, eucli_dist[0,1]]

def plot_roc(fpr, tpr, figure_name="roc.png"):

    plt.switch_backend('Agg')
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='#16a085',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#2c3e50', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", frameon=False)
    fig.savefig(figure_name, dpi=fig.dpi)
