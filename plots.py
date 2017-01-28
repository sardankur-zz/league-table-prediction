import itertools
import numpy as np
import matplotlib.pyplot as plt

#plot the confusion matrix
def plot_confusion_matrix(cm, classes, cmap,
                          normalize=False,title='Confusion matrix for match results'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()


def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')

# plotting team win stats for actual vs predicted
def plot_hist(dic1, dic2, chartname):
    fig, ax = plt.subplots()
    actual_wins = list(dic1.values())[:10]
    pred_wins = list(dic2.values())[:10]
    N = len(actual_wins)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.10  # the width of the bars
    max1 = max(actual_wins)
    min1 = min(actual_wins)
    max2 = max(pred_wins)
    min2 = min(pred_wins)
    plt.ylim([min(min1, min2), max(max1, max2) + 5])
    rects1 = ax.bar(ind, actual_wins, width, color='r')
    rects2 = ax.bar(ind + width, pred_wins, width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Wins')
    ax.set_title(chartname)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(dic1.keys())
    ax.legend((rects1[0], rects2[0]), ('Actual Number of Wins', 'predicted Number of Wins'))
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    plt.show()