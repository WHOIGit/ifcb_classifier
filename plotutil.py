#! /usr/bin/env python

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


def make_confusion_matrix_plot(true_labels, predict_labels, labels, title='Confusion matrix', normalize_mapping=True, text_as_percentage=False, output='show'):
    """
    Parameters:
        true_labels                  : These are your true classification categories
        predict_labels               : These are you predicted classification categories
        labels                       : This is a list of labels which will be used to display the axix labels
        title='Confusion matrix'     : Title for your matrix
        normalize=False              : assumedly, it normalizes the output for classes or different lengths

    Returns: a matplotlib confusion matrix figure
    inspired from https://stackoverflow.com/a/48030258
    """

    # sort labels alphabetically if none provided
    if labels is None:
        labels = sorted(list(set(true_labels)), reverse=True)

    sizes = dict(dpi=320, figsize=(6,6), h1=7, tick_label=3, cell=2)
    if len(labels) <= 10:
        sizes['figsize'] = (2,2)
        sizes['h1']      = 4
        sizes['cell']    = 2
        sizes['tick_labels'] = 7
    elif len(labels) <= 25:
        sizes['figsize'] = (3,3)
        sizes['h1']      = 5
        sizes['cell']    = 2.5
        sizes['tick_labels'] = 7
    elif len(labels) <= 50:
        sizes['figsize'] = (4,4)
        sizes['h1']      = 5
        sizes['cell']    = 2.5
        sizes['tick_labels'] = 7
    elif len(labels) <= 75:
        sizes['figsize']    = (5,5)
    elif len(labels) <=100:
        sizes['figsize']    = (5,5)
    else:
        #sizes['figsize']    = (6,6)
        sizes['cell']       = 1.9


    # creation of the matrix
    cm = confusion_matrix(true_labels, predict_labels, labels=labels)
    # normalized confusion matrix
    ncm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    ncm = np.nan_to_num(ncm, copy=True)

    # creation of the plot
    if not isinstance(output,mpl.axes.Axes):
        fig = plt.figure(figsize=sizes['figsize'], dpi=sizes['dpi'], facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = output
    ax.set_title(title, fontsize=sizes['h1'], loc='right')

    # adjusting color pallet to be not-so-dark
    cmap = mpl.cm.Oranges(np.arange(plt.cm.Oranges.N))
    cmap = mpl.colors.ListedColormap(cmap[0:-50])

    # applying color mapping and text formatting
    if normalize_mapping: im = ax.imshow(ncm, cmap=cmap)
    else:                 im = ax.imshow(cm, cmap=cmap)
    if text_as_percentage:
        cell_format = '{:.0f}%'
        cm = 100*ncm
    else: cell_format = '{:.0f}'

    # formatting labels (with column and row counts)
    label_format = '{} [{:>4}]'
    ylabels = [label_format.format(c, true_labels.count(c)) for c in labels]
    xlabels = [label_format.format(c, predict_labels.count(c)) for c in labels]

    ax.set_ylabel('True Input Classes', fontsize=sizes['h1'])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(ylabels, fontsize=sizes['tick_label'], va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    ax.set_xlabel('Predicted Output Classes', fontsize=sizes['h1'])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(xlabels, fontsize=sizes['tick_label'], rotation=90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    ax.tick_params(axis='x', which='minor', size=0)

    # adding cell text annotations
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cell_format.format(cm[i, j]) if cm[i, j] != 0 else '.',
                horizontalalignment='center', fontsize=sizes['cell'],
                verticalalignment='center', color='black')

    if not isinstance(output, mpl.axes.Axes):
        fig.set_tight_layout(True)

    # output
    if output=='show':
        plt.show()
    elif isinstance(output,str):
        #mpl.use('Agg')
        plt.savefig(output)

    return fig



def loss(train_eval_loss_tuples, output='show', title='', normalize=True):
    """ plot training and evaluation loss trends """
    train_loss, eval_loss = zip(*train_eval_loss_tuples)
    if normalize:
        train_loss = [loss/train_loss[0] for loss in train_loss]
        eval_loss  = [loss/eval_loss[0] for loss in eval_loss]
    epochs = range(len(train_eval_loss_tuples))

    if not isinstance(output,mpl.axes.Axes):
        fig = plt.figure(facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = output

    ax.set_title(title)
    ax.set_xlabel('Epochs')
    if normalize: ax.set_ylabel('Normalized Loss')
    else:         ax.set_ylabel('Loss')

    ax.plot(epochs,train_loss,'b-')
    ax.plot(epochs,eval_loss,'r-')
    ax.plot( eval_loss.index(min(eval_loss)), min(eval_loss), 'ro')

    # output
    if output=='show':
        plt.show()
    elif isinstance(output,str):
        plt.savefig(output)
    else:
        return ax


# TODO standalone cli functions
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("input", help="some input")
    args = parser.parse_args()  # ingest CLI arguments
    # do stuff
