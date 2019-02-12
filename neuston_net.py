#! /usr/bin/env python

## Builtin Imports ##
import os
import time
import itertools
from pprint import pprint
from ast import literal_eval

## 3rd party library imports ##
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
#from matplotlib.figure import Figure
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

## TORCH STUFF ##
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter
import torchvision.models as MODEL_MODULE

YMD_HMS = "%Y-%m-%d %H:%M:%S"

### UTILITIES ###

# average dimensions of images
from statistics import mean


def calc_avg_dimensions(imagefolder_dataset):
    sizes = [img[0].size for img in imagefolder_dataset]
    x, y = zip(*sizes)
    print('width avg:', mean(x))
    print('height avg:', mean(y))
    print('both avg:', mean(x+y))


#print('Training:')
#calc_avg_dimensions(datasets.ImageFolder(root=train_root))
#print('Testing:')
#calc_avg_dimensions(datasets.ImageFolder(root=test_root))

# only had to do this once. this was with 50:50 data
results = '''
Training:
    width avg: 243.51
    height avg: 85.19
    both avg:  164.35
Testing:
    width avg: 243.81
    height avg: 85.80
    both avg:  164.81
'''


def training_progressbar(batch: int, batches: int, epoch: int, epochs: int,
                         batches_per_dot: int = 1, batches_per_summary: int = 70):
    if batch%batches_per_summary == batches_per_summary-1 or batch == 0:
        phase_percent_done = 100*batch/batches
        total_percent_done = 100*(batches*epoch+batch)/(batches*epochs)
        print('\n[ {:1.1f} %] epoch {:<2} Training {:1.1f}% complete'.format(total_percent_done, epoch+1,
                                                                             phase_percent_done), end=' ', flush=True)
    if batch%batches_per_dot == batches_per_dot-1:
        print('.', end='', flush=True)


def testing_progressbar(batch: int, batches: int, batches_per_dot: int = 1, batches_per_summary: int = 85):
    if batch%batches_per_summary == batches_per_summary-1 or batch == 0:
        phase_percent_done = 100*batch/batches
        print('\nVerification {:1.1f}% complete'.format(phase_percent_done), end=' ', flush=True)

    if batch%batches_per_dot == batches_per_dot-1:
        print('.', end='', flush=True)


def ts2secs(ts1, ts2):
    #_start_time = time.strftime(YMD_HMS)
    #_elapsed_secs = ts2secs(_start_clock, _train_clock)
    t2 = time.mktime(time.strptime(ts2, YMD_HMS))
    t1 = time.mktime(time.strptime(ts1, YMD_HMS))
    return t2-t1


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    Example setup:      dataloader = torch.utils.DataLoader(ImageFolderWithPaths("path/to/your/perclass/image/folders"))
    Example usage:     for inputs,labels,paths in my_dataloader: ....
    instead of:        for inputs,labels in my_dataloader: ....
    adapted from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple+(path,))
        return tuple_with_path


def make_confusion_matrix_plot(true_labels, predict_labels, labels, title='Confusion matrix', normalize_mapping=True,
                               text_as_percentage=False, show=True, outfile=None):
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

    # creation of the matrix
    cm = confusion_matrix(true_labels, predict_labels, labels=labels)
    # normalized confusion matrix
    ncm = 100*cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    ncm = np.nan_to_num(ncm, copy=True)

    # creation of the plot
    fig = plt.figure(figsize=(6, 6), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=7, loc='right')

    # adjusting color pallet to be not-so-dark
    cmap = mpl.cm.Oranges(np.arange(plt.cm.Oranges.N))
    cmap = mpl.colors.ListedColormap(cmap[0:-50])

    # applying color mapping and text formatting
    if normalize_mapping:
        im = ax.imshow(ncm, cmap=cmap)
    else:
        im = ax.imshow(cm, cmap=cmap)
    if text_as_percentage:
        cell_format = '{:.1f}%'
        cm = ncm
    else:
        cell_format = '{:.0f}'

    # formatting labels (with column and row counts)
    label_format = '{} [{:>4}]'
    ylabels = [label_format.format(c, true_labels.count(c)) for c in labels]
    xlabels = [label_format.format(c, predict_labels.count(c)) for c in labels]

    ax.set_ylabel('True Input Classes', fontsize=7)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(ylabels, fontsize=3, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    ax.set_xlabel('Predicted Output Classes', fontsize=7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(xlabels, fontsize=3, rotation=90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # adding cell text annotations
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cell_format.format(cm[i, j]) if cm[i, j] != 0 else '.',
                horizontalalignment='center', fontsize=1.9,
                verticalalignment='center', color='black')

    fig.set_tight_layout(True)

    # output
    if show:
        plt.show()
    if outfile:
        plt.savefig(outfile)

    return fig


def record_epoch_stats(epoch, verification_dict, name, savepath, filename, secs_elapsed, append=False):
    verification_dict['name'] = name
    verification_dict['epoch'] = epoch
    verification_dict['secs_elapsed'] = secs_elapsed

    fpath = '{}/{}'.format(savepath, filename)
    #print('writing to', fpath, '... ', end='')

    if append:
        # read content of existing file if available
        verification_dict.pop('prediction_ranks')
        verification_dict.pop('true_images')
        try:
            with open(fpath, 'r') as f:
                epoch_records = literal_eval(f.read())
        except FileNotFoundError:
            epoch_records = []

        # append per-epoch info into dict
        epoch_records.append(verification_dict)

        # overwrite-save  records
        with open(fpath, 'w') as f:
            print(epoch_records, file=f)

    else:
        with open(fpath, 'w') as f:
            print(verification_dict, file=f)


## TRAINING AND VALIDATING ##


def training_loop(model, training_loader, testing_loader, epochs, device, epoch0=0, optimizer=None,
                  savepath=None, logger=None, best_loss=float('inf'), normalize_weights=False):
    run_name = os.path.basename(savepath)
    num_o_batches = len(training_loader)
    best_epoch = epoch0
    best_f1 = 0
    print('Training... there are {} batches per epoch'.format(num_o_batches))
    print('Starting at epoch {} of {}'.format(epoch0+1, epochs))

    # defining loss function
    try:
        dataset = training_loader.dataset.datasets[0]
    except AttributeError:
        dataset = training_loader.dataset
    if normalize_weights:
        weights = [0 for c in dataset.classes]
        for img, img_idx in dataset.imgs:
            weights[img_idx] += 1
        weights = [len(dataset.imgs)/len(weights)/weight for weight in weights]
        weights = torch.FloatTensor(weights).to(device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weights).to(device)

    training_startime = time.strftime(YMD_HMS)
    for epoch in range(epoch0+1, epochs+1):
        model.train()
        epoch_startime = time.strftime(YMD_HMS)
        epoch_loss = 0

        #Process data batch
        for i, data in enumerate(training_loader):
            # send data to gpu(s)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # run model and determin loss
            try:
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels)
            except AttributeError:  # e="'tuple' object has no attribute 'to'" when inception_v3 aux_logits=True
                outputs, aux_outputs = model(inputs)
                outputs, aux_outputs = outputs.to(device), aux_outputs.to(device)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1+0.4*loss2

            # train model
            loss.backward()
            optimizer.step()

            # batch stats
            epoch_loss += loss.item()

            # batch LOGGING
            #logger.add_scalar('TRAINING Loss', loss, num_o_batches*epoch+i)
            training_progressbar(i, num_o_batches, epoch-1, epochs, 2, 140)

        ## per epoch VERIFICATION ##
        verified = verification_loop(model, testing_loader, device, weights)
        # verified: validation_loss      classes
        #           true_inputs          prediction_outputs
        #           true_images          prediction_ranks

        ## BOOKKEEPING ##
        classes = verified['classes']
        input_labels = [classes[c] for c in verified['true_inputs']]
        output_labels = [classes[c] for c in verified['prediction_outputs']]
        epoch_overall_f1_weighted = f1_score(verified['true_inputs'], verified['prediction_outputs'], average='weighted')
        epoch_overall_f1_macro = f1_score(verified['true_inputs'], verified['prediction_outputs'], average='macro')
        f1_perclass = f1_score(input_labels, output_labels, average=None)
        recall_perclass = recall_score(input_labels, output_labels, average=None)

        classes_by_recall = sorted(verified['classes'], reverse=True,
                                   key=lambda c: (recall_perclass[classes.index(c)], f1_perclass[classes.index(c)]))

        # Saving the model if it's the best
        if savepath and verified['validation_loss'] < best_loss:
            # saving model
            torch.save(model.state_dict(), '{}/model.state'.format(savepath))
            # Later to restore for usage (not training):
            #model.load_state_dict(torch.load(filepath))
            #model.eval()

            # To save to continue training...
            #state = dict(epoch=epoch,
            #             model=model.state_dict(),
            #             optimizer=optimizer.state_dict(),
            #             validation_loss=verified['validation_loss'])
            #torch.save(state, '{}/model.state'.format(savepath))

            # saving training results
            record_epoch_stats(epoch=epoch, verification_dict=verified, append=False, name=run_name,
                               savepath=savepath, filename='best_training_result.dict',
                               secs_elapsed=ts2secs(training_startime, time.strftime(YMD_HMS)))

            # saving easy-to-read file
            quickstats = dict(name=run_name, epoch=epoch, loss=round(verified['validation_loss'], 1),
                              f1_weighted=round(epoch_overall_f1_weighted, 3),
                              f1_macro=round(epoch_overall_f1_macro, 3),
                              hours_elapsed=round(ts2secs(training_startime, time.strftime(YMD_HMS))/60/60, 2))
            with open('{}/{}'.format(savepath, 'quickstats.txt'), 'w') as f:
                pprint(quickstats, f)

            # confusion matrix
            title = '{} Confusion Matrix (f1_w={:.1f}% f1_m={:.1f}%)'.format(run_name, 100*epoch_overall_f1_weighted,
                                                                             100*epoch_overall_f1_macro)
            cm_figure = make_confusion_matrix_plot(input_labels, output_labels, labels=classes_by_recall,
                                                   title=title, normalize_mapping=True, text_as_percentage=True)
            cm_figure.savefig('{}/{}'.format(savepath, 'confusion_matrix.png'))

        # Epoch Logging
        if savepath:
            record_epoch_stats(epoch=epoch, verification_dict=verified, append=True,
                               savepath=savepath, filename='all_training_results.lod',
                               name=run_name,
                               secs_elapsed=ts2secs(training_startime, time.strftime(YMD_HMS)))

        # epoch LOGGING with tensorboardx
        if logger:
            try:
                firstloss_training == firstloss_verification  # any statement really
            except UnboundLocalError:
                firstloss_training = epoch_loss
                firstloss_verification = verified['validation_loss']

            normalized_training_loss = 100*epoch_loss/firstloss_training
            normalized_verification_loss = 100*verified['validation_loss']/firstloss_verification

            logger.add_scalars('Epoch loss', {'Training':     epoch_loss,
                                              'Verification': verified['validation_loss']}, epoch)
            logger.add_scalars('Normalized Epoch loss', {'Training':     normalized_training_loss,
                                                         'Verification': normalized_verification_loss}, epoch)
            logger.add_scalars('F1 per class',
                               f1_score(verified['true_inputs'], verified['prediction_outputs'], average=None), epoch)
            logger.add_scalars('Overall Metrics', {'F1_weighted':        epoch_overall_f1_weighted,
                                                   'recall_weighted':    recall_score(verified['true_inputs'],
                                                                                      verified['prediction_outputs'],
                                                                                      average='weighted'),
                                                   'precision_weighted': precision_score(verified['true_inputs'],
                                                                                         verified['prediction_outputs'],
                                                                                         average='weighted')})
            title = '{} Confusion Matrix (f1_w={:.1f}% f1_m={:.1f}%)'.format(run_name, 100*epoch_overall_f1_weighted,
                                                                             100*epoch_overall_f1_macro)
            cm_figure = make_confusion_matrix_plot(input_labels, output_labels, labels=classes_by_recall,
                                                   title=title, normalize_mapping=True, text_as_percentage=True)
            logger.add_figure('Confusion Matrix', cm_figure, epoch)

        # terminal output
        now = time.strftime(YMD_HMS)
        epoch_elapsed = ts2secs(epoch_startime, now)
        total_elapsed = ts2secs(training_startime, now)
        output_string = 'Epoch {} done in {:.0f} mins. loss={:.3f}. f1_w={:.1f}% f1_m={:.1f}%. Total elapsed time={:.2f} hrs'
        print(output_string.format(epoch, epoch_elapsed/60, verified['validation_loss'],
                                   100*epoch_overall_f1_weighted, 100*epoch_overall_f1_macro, total_elapsed/(60*60)))

        # note or record best epoch, or stop early
        if best_loss > verified['validation_loss']:
            best_loss = verified['validation_loss']
            best_epoch = epoch
            best_f1 = epoch_overall_f1_weighted
        elif epoch > 1.33*best_epoch and epoch >= 10:
            print("Model probably won't be improving more from here, shutting this operation down")
            print('Best Epoch was {} with a validation loss of {:.3f} and an F1 score of {:.1f}%'.format(best_epoch,
                                                                                                         best_loss,
                                                                                                         100*best_f1))
            return
        if best_epoch != epoch:
            print('Best Epoch was {} with a validation loss of {:.3f} and an F1 score of {:.1f}%'.format(best_epoch,
                                                                                                         best_loss,
                                                                                                         100*best_f1))


def verification_loop(model, testing_loader, device, weights=None):
    """ returns dict of: accuracy, loss, perclass_accuracy, perclass_correct, perclass_totals,
                         true_inputs, predicted_outputs, f1_avg, perclass_f1, classes )"""
    try:
        classes = testing_loader.dataset.classes
    except AttributeError:
        classes = testing_loader.dataset.datasets[0].classes

    sum_loss = 0
    all_true_input_labels, all_predicted_output_labels = [], []
    all_predicted_output_ranks = []
    all_input_image_paths = []

    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(weights).to(device)
        for i, data in enumerate(testing_loader):

            # forward pass
            input_images, true_input_labels, input_image_paths = data
            input_images, true_input_labels = input_images.to(device), true_input_labels.to(device)

            # run model and determin loss
            try:
                outputs = model(input_images).to(device)
                loss = criterion(outputs, true_input_labels)
            except AttributeError:  # e="'tuple' object has no attribute 'to'" when inception_v3 aux_logits=True
                outputs, aux_outputs = model(input_images)
                outputs, aux_outputs = outputs.to(device), aux_outputs.to(device)
                loss1 = criterion(outputs, true_input_labels)
                loss2 = criterion(aux_outputs, true_input_labels)
                loss = loss1+0.4*loss2

            sum_loss += loss.item()

            # format results
            predicted_output_ranks, predicted_output_labels = torch.max(outputs, 1)
            # outputs format is a tensor list of lists, inside lists are len(classes) long,
            # each with a per-class weight prediction. outide list is batch_size long.
            # doing torch.max returns the index of the highest per-class prediction (inside list),
            # for all elements of outside list. Therefor len(predicted)==len(output)==batch_size

            all_predicted_output_labels.extend([p.item() for p in predicted_output_labels])
            all_predicted_output_ranks.extend([r.item() for r in predicted_output_ranks])
            all_true_input_labels.extend([t.item() for t in true_input_labels])
            all_input_image_paths.extend(input_image_paths)

            # printing progress
            testing_progressbar(i, len(testing_loader))

    # result metrics
    assert len(all_predicted_output_labels) == \
           len(all_predicted_output_ranks) == \
           len(all_true_input_labels) == \
           len(all_input_image_paths)
    validation_dict = dict(validation_loss=sum_loss,
                           classes=classes,
                           true_inputs=all_true_input_labels,
                           prediction_outputs=all_predicted_output_labels,
                           prediction_ranks=all_predicted_output_ranks,
                           true_images=all_input_image_paths)
    '''
    misprediction_metadata = {}
    classes = verified['classes']
    combo = zip(verified['true_inputs'],verified['true_images'],verified['prediction_outputs'],verified['prediction_ranks'])
    for input_label,input_image,output_label,output_rank in combo:
        if input_label == output_label: continue

        a_mistake = dict(predicted=classes[output_label], rank=output_rank, image=input_image)
        try: misprediction_metadata[classes[input_label]].append(a_mistake)
        except KeyError:
            misprediction_metadata[classes[input_label]] = [a_mistake]
    '''
    return validation_dict


## INITIALIZATION ##

import subprocess, argparse

if __name__ == '__main__':
    model_choices = ['inception_v3', 'alexnet', 'squeezenet',
                     'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet151',
                     'densenet121', 'densenet169', 'densenet161', 'densenet201']
    augmentation_choices = ['flipx', 'flipy', 'flipxy', 'training-set-only', 'chance']

    ## Parsing command line input ##
    parser = argparse.ArgumentParser()
    parser.add_argument("training_dir", help="path to training set images")
    parser.add_argument("testing_dir", help="path to testing set images")
    parser.add_argument("output_dir", help="directory out output logs and saved model")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help='Preloads model with weights trained on imagenet')
    parser.add_argument("--normalize", default=False, action='store_true',
                        help='Classes will be weighted such that classes with fewer instances will train better')
    parser.add_argument("--epochs", default=10, type=int,
                        help="How many epochs to run (default 10).\nTraining may end before <epochs> if validation loss keeps rising beyond best_epoch*1.33")
    parser.add_argument("--batch-size", default=108, type=int, dest='batch_size',
                        help="how many images to process in a batch, (default 108, a number divisible by 1,2,3,4 to ensure even batch distribution across up to 4 GPUs)")
    parser.add_argument("--model", default='inception_v3', choices=model_choices,
                        help="select a model architecture to train (default to inceptin_v3)")
    parser.add_argument("--loaders", default=4, type=int,
                        help='number of threads to use for loading data to/from GPUs. 4 per GPU is good. (Default 4)')
    parser.add_argument("--tensorboard-logging", default=False, action='store_true',
                        help='logs training/validation metrics in tensorboard folder/format')
    parser.add_argument("--augment", default=[], nargs='+', choices=augmentation_choices, help='''Data augmentation. Adds transformed images to the non-transformed dataset*.
    flipx and flipy denotes mirroring the image vertically and horizontally respectively. To have images flipped along both axis, include flipxy as well (unless "chance" is designated).
    *: if "chance" is included, transformed images are not added to the base dataset; instead, each time an image is loaded to memory, transformations may be applied.
    If "training-set-only" is included, augmentation transformations will only be applied to the training set.''')
    #parser.add_argument("--freeze-layers", default=None, type=int,
    #                    help='freezes layers 0-n of model. assumes --pretrained or --resume. (default None)')
    #parser.add_argument("--resume", default=False, nargs='?', const=True,
    #                    help="path to a previously saved model to pick up from (defaults to <output_dir>)")
    #parser.add_argument("--image-size",default=224,type=int, dest='image_size',help="scales images to n-by-n pixels (default: 224)")
    #parser.add_argument("--no-save",default=False,action="store_true",dest="no_save", help="if True, a model will not be saved under <output_dir> (default: False)")
    #parser.add_argument("--letterbox",default=False,action='store_true')
    # see https://github.com/laurent-dinh/pylearn/blob/master/pylearn2/utils/image.py find:letterbox
    #parser.add_argument("--verbose",'-v', action="store_true", help="printout epoch dot progressions", default=False)
    #parser.add_argument("--config", help="path to specific ini config file for model and transforms transforms, (default invception_v3)")
    args = parser.parse_args()  # ingest CLI arguments

    # input cleanup
    if not args.output_dir.startswith('output/'):
        args.output_dir = 'output/'+args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not args.training_dir.startswith('data/'):
        args.training_dir = 'data/'+args.training_dir
    if not args.testing_dir.startswith('data/'):
        args.testing_dir = 'data/'+args.testing_dir

    print("Output directory: {}".format(args.output_dir))
    print("pyTorch VERSION:", torch.__version__)

    ## gpu torch setup ##
    if torch.cuda.is_available():
        gpus = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        # pytorch keys off the ~index~ of CUDA_VISIBLE_DEVICES.
        # So if gpus == [3,4] then device "cuda:0" == GPU no. 3
        #                      and device "cuda:1" == GPU no. 4
        device = torch.device('cuda:0')
        currdev = torch.cuda.current_device()
    else:
        gpus = []
        device = torch.device("cpu")
        currdev = None
    print("CUDA_VISIBLE_DEVICES: {}".format(gpus))
    print("Active Device is {} (aka torch.cuda.current_device() == {} )".format(device, currdev))

    ## data info ##
    training_numOfiles = subprocess.check_output('find {} -type f | wc -l'.format(args.training_dir),
                                                 shell=True).decode().strip()
    testing_numOfiles = subprocess.check_output('find {} -type f | wc -l'.format(args.testing_dir),
                                                shell=True).decode().strip()
    training_dirSize = subprocess.check_output('du --apparent-size -sh {}'.format(args.training_dir),
                                               shell=True).decode().strip()
    testing_dirSize = subprocess.check_output('du --apparent-size -sh {}'.format(args.testing_dir),
                                              shell=True).decode().strip()
    print(
        "Un-augmented Training Set: {} ({} files, {})".format(args.training_dir, training_numOfiles, training_dirSize))
    print("Un-augmented Testing Set: {} ({} files, {})".format(args.testing_dir, testing_numOfiles, testing_dirSize))

    # setting model appropriate resize
    if args.model == 'inception_v3':
        pixels = [299, 299]
    else:
        pixels = [224, 224]

    ## initializing data ##
    base_tforms = [transforms.Resize(pixels),  # 299x299 is minimum size for inception
                   transforms.ToTensor()]

    # augmentation #
    if 'chance' in args.augment:
        # when data is loaded into model, there is a 50% chance per axis it will be transformed
        aug_tforms = []
        if any([arg in args.augment for arg in ['flip', 'flipxy', 'flip-xy', 'flip-both']]):
            aug_tforms.append(transforms.RandomVerticalFlip(p=0.5))
            aug_tforms.append(transforms.RandomHorizontalFlip(p=0.5))
        elif any([arg in args.augment for arg in ['flipx', 'flip-x', 'flip-vertical']]):
            aug_tforms.append(transforms.RandomVerticalFlip(p=0.5))
        elif any([arg in args.augment for arg in ['flipy', 'flip-y', 'flip-horizontal']]):
            aug_tforms.append(transforms.RandomHorizontalFlip(p=0.5))

        tforms = transforms.Compose(aug_tforms+base_tforms)
        training_dataset = datasets.ImageFolder(root=args.training_dir, transform=tforms)
        if 'training-set-only' in args.augment:
            tforms = transforms.Compose(base_tforms)
        validation_dataset = ImageFolderWithPaths(root=args.testing_dir, transform=tforms)

    elif args.augment != []:
        # when data is initialized, a translated copies of it will be initialized too
        tforms_lol = [base_tforms]
        if any([arg in args.augment for arg in ['flipx', 'flip-x', 'flip-vertical']]):
            tforms_lol.append([transforms.RandomVerticalFlip(p=1)]+base_tforms)
        if any([arg in args.augment for arg in ['flipy', 'flip-y', 'flip-horizontal']]):
            tforms_lol.append([transforms.RandomHorizontalFlip(p=1)]+base_tforms)
        if any([arg in args.augment for arg in ['flip', 'flipxy', 'flip-xy', 'flip-both']]):
            tforms_lol.append([transforms.RandomVerticalFlip(p=1),
                               transforms.RandomHorizontalFlip(p=1)]+base_tforms)
        training_datasets, validation_datasets = [], []
        for tforms in tforms_lol:
            tforms = transforms.Compose(tforms)
            training_datasets.append(datasets.ImageFolder(root=args.training_dir, transform=tforms))
            validation_datasets.append(ImageFolderWithPaths(root=args.testing_dir, transform=tforms))
        training_dataset = torch.utils.data.ConcatDataset(training_datasets)
        if 'training-set-only' in args.augment:
            tforms = transforms.Compose(base_tforms)
            validation_dataset = ImageFolderWithPaths(root=args.testing_dir, transform=tforms)
        else:
            validation_dataset = torch.utils.data.ConcatDataset(validation_datasets)
        tforms = tforms_lol  # for cli printint later
    # no augmentation #
    else:
        tforms = transforms.Compose(base_tforms)
        training_dataset = datasets.ImageFolder(root=args.training_dir, transform=tforms)
        validation_dataset = ImageFolderWithPaths(root=args.testing_dir, transform=tforms)

    # create dataloaders
    print('Loading Training Data...')
    training_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True, batch_size=args.batch_size,
                                                  pin_memory=True, num_workers=args.loaders)
    print('Loading Validation Data...')
    testing_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=True, batch_size=args.batch_size,
                                                 pin_memory=True, num_workers=args.loaders)

    # cli feedback
    training_len, testing_len = len(training_loader.dataset), len(testing_loader.dataset)
    try:
        assert training_loader.dataset.classes == testing_loader.dataset.classes
        num_o_classes = len(training_loader.dataset.classes)
    except AttributeError:
        assert training_loader.dataset.datasets[0].classes == testing_loader.dataset.datasets[0].classes
        num_o_classes = len(training_loader.dataset.datasets[0].classes)
    print("Number of Classes:", num_o_classes)
    if args.augment:
        print('Augmented Training datapoints {}\nAugmented Testing datapoints {}'.format(training_len, testing_len))

    ## initializing model model  ##
    # options:
    #     inception_v3, alexnet, squeezenet,
    #     vgg11, vgg13, vgg16, vgg19,
    #     vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
    #     resnet18, resnet34, resnet50, resnet101, resnet151,
    #     densenet121, densenet169, densenet161, densenet201

    if args.model == 'inception_v3':
        model = MODEL_MODULE.inception_v3(args.pretrained)  #, num_classes=num_o_classes, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_o_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_o_classes)
    elif args.model == 'alexnet':
        model = getattr(MODEL_MODULE, args.model)(args.pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_o_classes)
    elif args.model == 'squeezenet':
        model = getattr(MODEL_MODULE, args.model+'1_1')(args.pretrained)
        model.classifier[1] = nn.Conv2d(512, num_o_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_o_classes
    elif args.model.startswith('vgg'):
        model = getattr(MODEL_MODULE, args.model)(args.pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_o_classes)
    elif args.model.startswith('resnet'):
        model = getattr(MODEL_MODULE, args.model)(args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_o_classes)
    elif args.model.startswith('densenet'):
        model = getattr(MODEL_MODULE, args.model)(args.pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_o_classes)

    # freeze layers - probably still works, uncomment here.
    # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
    #try:
    #    num_o_layers = len(list(model.children()))
    #    for i, child in enumerate(model.children()):
    #        if i > args.freeze_layers: break
    #        print('Freezing layer [{}/{}]: {}'.format(i, num_o_layers, child.__class__))
    #        for param in child.parameters():
    #            param.requires_grad = False
    #except TypeError as e:
    #    if 'NoneType' in str(e):
    #        print(args.freeze_layers);
    #        pass  # for when args.freeze_layers == None
    #    else:
    #        raise e

    # multiple GPU wrapper
    if torch.cuda.device_count() > 1:  # if multiple-gpu's detected, use available gpu's
        model = nn.DataParallel(model, device_ids=list(range(len(gpus))))
    model.to(device)  # sends model to GPU

    # optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

    epoch0, best_loss = 0, float('inf')
    # deactivated because unused and untested
    # load a previously saved model if one exists, if --resume enabled, otherwise use defaults
    #print('RESUME:', args.resume)
    #if isinstance(args.resume, str):
    #    model_loadpath = args.resume
    #else:
    #    model_loadpath = '{}/model.state'.format(args.output_dir)
    #if args.resume:
    #    try:
    #        savestate = torch.load(model_loadpath)
    #        model.load_state_dict(savestate['model'])
    #        optimizer.load_state_dict(savestate['optimizer'])
    #        epoch0 = savestate['epoch']
    #        best_loss = savestate['validation_loss']
    #        print('loading saved model:', model_loadpath)
    #    except FileNotFoundError as e:
    #        print(e)

    print("Model: {}".format(model.__class__))
    print("Transformations: {}".format(tforms))
    print("Epochs: {}, Batch size: {}".format(args.epochs, args.batch_size))

    ## initializing logger ##
    if args.tensorboard_logging:
        logger = SummaryWriter(os.path.join(args.output_dir, 'tboard_logs'))
    else:
        logger = None

    training_loop(model, training_loader, testing_loader, args.epochs, device, epoch0=epoch0, logger=logger,
                  optimizer=optimizer, savepath=args.output_dir, best_loss=best_loss, normalize_weights=args.normalize)
