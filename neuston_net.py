#! /usr/bin/env python

## Builtin Imports ##
import os
import sys
import time
import random
from ast import literal_eval
import warnings

## 3rd party library imports ##
import pandas as pd
from sklearn import metrics

## TORCH STUFF ##
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torchvision.models as MODEL_MODULE
import torchvision

## local imports ##
try: import plotutil  # installation of matplotlib is optional
except ImportError: plotutil = None

YMD_HMS = "%Y-%m-%d %H:%M:%S"
warnings.filterwarnings("ignore",
    category=metrics.classification.UndefinedMetricWarning)

### UTILITIES ###

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


class NeustonDataset(Dataset):

    def __init__(self, src, minimum_images_per_class=1, transforms=None, images_perclass=None):
        self.src = src
        if not images_perclass:
            images_perclass = self.fetch_images_perclass(src)

        self.minimum_images_per_class = max(1, minimum_images_per_class)  # always at least 1.
        new_images_perclass = {label: images for label, images in images_perclass.items() if
                               len(images) >= self.minimum_images_per_class}
        classes_ignored = sorted(set(images_perclass.keys())-set(new_images_perclass.keys()))
        self.classes_ignored_from_too_few_samples = [(c, len(images_perclass[c])) for c in classes_ignored]
        self.classes = sorted(new_images_perclass.keys())

        # flatten images_perclass to congruous list of image paths and target id's
        self.targets, self.images = zip(*((self.classes.index(t), i) for t in new_images_perclass for i in new_images_perclass[t]))
        self.transforms = transforms

    @staticmethod
    def fetch_images_perclass(src):
        """ folders in src are the classes """
        classes = [d.name for d in os.scandir(src) if d.is_dir()]
        classes.sort()

        images_perclass = {}
        for subdir in classes:
            files = os.listdir(os.path.join(src, subdir))
            #files = sorted([i for i in files if i.lower().endswith(ext)])
            files = sorted([f for f in files if os.path.splitext(f)[1] in datasets.folder.IMG_EXTENSIONS])
            images_perclass[subdir] = [os.path.join(src, subdir, i) for i in files]
        return images_perclass

    @property
    def images_perclass(self):
        ipc = {c: [] for c in self.classes}
        for img, trg in zip(self.images, self.targets):
            ipc[self.classes[trg]].append(img)
        return ipc

    def split(self, ratio1, ratio2, seed=None, minimum_images_per_class='scale'):
        assert ratio1+ratio2 == 100
        d1_perclass = {}
        d2_perclass = {}
        for class_label, images in self.images_perclass.items():
            #1) determine output lengths
            d1_len = int(ratio1*len(images)/100+0.5)
            d2_len = len(images)-d1_len  # not strictly needed

            #2) split images as per distribution
            if seed:
                random.seed(seed)
            d1_images = random.sample(images, d1_len)
            d2_images = sorted(list(set(images)-set(d1_images)))
            assert len(d1_images)+len(d2_images) == len(images)

            #3) put images into perclass_sets at the right class
            d1_perclass[class_label] = d1_images
            d2_perclass[class_label] = d2_images

        #4) calculate minimum_images_per_class for thresholding
        if minimum_images_per_class == 'scale':
            d1_threshold = int(self.minimum_images_per_class*ratio1/100+0.5)
            d2_threshold = self.minimum_images_per_class-d1_threshold
        elif isinstance(minimum_images_per_class,int):
            d1_threshold = d2_threshold = minimum_images_per_class
        else:
            d1_threshold = d2_threshold = self.minimum_images_per_class

        #5) create and return new datasets
        dataset1 = NeustonDataset(src=self.src, images_perclass=d1_perclass, transforms=self.transforms, minimum_images_per_class=d1_threshold)
        dataset2 = NeustonDataset(src=self.src, images_perclass=d2_perclass, transforms=self.transforms, minimum_images_per_class=d2_threshold)
        assert dataset1.classes == dataset2.classes  # possibly fails due to edge case thresholding?
        assert len(dataset1)+len(dataset2) == len(self)  # make sure we don't lose any images somewhere
        return dataset1, dataset2

    @classmethod
    def from_csv(cls, src, csv_file, column_to_run, transforms=None, minimum_images_per_class=None):
        #1) load csv
        df = pd.read_csv(csv_file, header=0)
        base_list = df.iloc[:,0].tolist()      # first column
        mod_list = df[column_to_run].tolist()  # chosen column

        #2) get list of files
        default_images_perclass = cls.fetch_images_perclass(src)
        missing_classes_src = [c for c in default_images_perclass if c not in base_list]

        #3) for classes in column to run, keep 1's, dump 0's, combine named
        new_images_perclass = {}
        missing_classes_csv = []
        skipped_classes = []
        grouped_classes = {}
        for base, mod in zip(base_list, mod_list):
            if base not in default_images_perclass:
                missing_classes_csv.append(base)
                continue

            if str(mod) == '0':  # don't include this class
                skipped_classes.append(base)
                continue
            elif str(mod) == '1':
                class_label = base  # include this class
            else:
                class_label = mod  # rename/group base class as mod
                if mod not in grouped_classes:
                    grouped_classes[mod] = [base]
                else:
                    grouped_classes[mod].append(base)

            # transcribing images
            if class_label not in new_images_perclass:
                new_images_perclass[class_label] = default_images_perclass[base]
            else:
                new_images_perclass[class_label].extend(default_images_perclass[base])

        #4) print messages
        if missing_classes_src:
            msg = '\n{} of {} classes from src dir {} were NOT FOUND in {}'
            msg = msg.format(len(missing_classes_src), len(default_images_perclass.keys()), src,
                             os.path.basename(csv_file))
            print('\n    '.join([msg]+missing_classes_src))

        if missing_classes_csv:
            msg = '\n{} of {} classes from {} were NOT FOUND in src dir {}'
            msg = msg.format(len(missing_classes_csv), len(base_list), os.path.basename(csv_file), src)
            print('\n    '.join([msg]+missing_classes_csv))

        if grouped_classes:
            msg = '\n{} GROUPED classes were created, as per {}'
            msg = msg.format(len(grouped_classes), os.path.basename(csv_file))
            print(msg)
            for mod, base_entries in grouped_classes.items():
                print('  {}'.format(mod))
                msgs = '     <-- {}'
                msgs = [msgs.format(c) for c in base_entries]
                print('\n'.join(msgs))

        if skipped_classes:
            msg = '\n{} classes were SKIPPED, as per {}'
            msg = msg.format(len(skipped_classes), os.path.basename(csv_file))
            print('\n    '.join([msg]+skipped_classes))

        #5) create dataset
        return cls(src=src, images_perclass=new_images_perclass, transforms=transforms, minimum_images_per_class=minimum_images_per_class)

    def __getitem__(self, index):
        path = self.images[index]
        target = self.targets[index]
        data = datasets.folder.default_loader(path)
        if self.transforms is not None:
            data = self.transforms(data)
        return data, target, path

    def __len__(self):
        return len(self.images)

    @property
    def imgs(self):
        return self.images


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
        data, target = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # return a new tuple that includes original plus the path
        return data, target, path


def record_epoch_stats(epoch, eval_dict, training_loss, cli_args, name, savepath, filename, secs_elapsed, append=False):
    save_dict = eval_dict.copy()
    save_dict['name'] = name
    save_dict['epoch'] = epoch
    save_dict['secs_elapsed'] = secs_elapsed
    save_dict['training_loss'] = training_loss
    save_dict.update(cli_args)

    fpath = os.path.join(savepath, filename)
    #print('writing to', fpath, '... ', end='')

    if append:
        # TODO CHANGE from reading in and rewriting previous file to inserting ",{save_dict}" at the nth-1 character position of the file (ie right before the closing ] )
        # read content of existing file if available
        try:
            with open(fpath, 'r') as f:
                epoch_records = literal_eval(f.read())
        except FileNotFoundError:
            epoch_records = []

        # append per-epoch info into dict
        epoch_records.append(save_dict)

        # overwrite-save  records
        with open(fpath, 'w') as f:
            print(epoch_records, file=f)

    else:
        with open(fpath, 'w') as f:
            print(save_dict, file=f)


## TRAINING AND VALIDATING ##

def training_loop(model, training_loader, eval_loader, device, savepath, optimizer, criterion,
                  weights=None, max_epochs=100, min_epochs=10, epoch0=0, best_loss=float('inf'), cli_args={}):
    run_name = os.path.basename(savepath)
    num_o_batches = len(training_loader)
    best_epoch = epoch0
    best_f1 = 0
    loss_record = []  # with one train-eval loss tuple per epoch

    print('Training... there are {} batches per epoch'.format(num_o_batches))
    #print('Starting at epoch {} of max {}'.format(epoch0+1, max_epochs))

    # start training loop
    training_startime = time.strftime(YMD_HMS)
    for epoch in range(epoch0+1, max_epochs+1):
        model.train()
        epoch_startime = time.strftime(YMD_HMS)
        epoch_loss = 0

        #Process data batch
        for i, data in enumerate(training_loader):
            # send data to gpu(s)
            inputs, labels, _ = data  # _ is for paths, not needed for training
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # run model and determine loss
            outputs = model(inputs)  #training-loop
            try:
                outputs = outputs.to(device)
                batch_loss = criterion(outputs, labels)
            except AttributeError:  # e="'tuple' object has no attribute 'to'" when inception_v3 aux_logits=True
                outputs, aux_outputs = outputs
                outputs, aux_outputs = outputs.to(device), aux_outputs.to(device)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                batch_loss = loss1+0.4*loss2

            # train model
            batch_loss.backward()
            optimizer.step()

            # batch stats
            epoch_loss += batch_loss.item()

            # batch LOGGING
            #logger.add_scalar('TRAINING Loss', loss, num_o_batches*epoch+i)
            training_progressbar(i, num_o_batches, epoch-1, max_epochs, 2, 140)

        ## per epoch EVALUATION ##
        assay = eval_loop(model, eval_loader, device, criterion)
        # verified: eval_loss   (float)
        #           classes (list of strings)
        #           true_images (images)
        #           true_inputs (satiations)
        #           prediction_outputs (satiations)

        ## BOOKKEEPING ##
        classes = assay['classes']
        input_labels = [classes[c] for c in assay['true_inputs']]
        output_labels = [classes[c] for c in assay['prediction_outputs']]
        f1_weighted = metrics.f1_score(assay['true_inputs'], assay['prediction_outputs'], average='weighted')
        f1_macro = metrics.f1_score(assay['true_inputs'], assay['prediction_outputs'], average='macro')
        f1_perclass = metrics.f1_score(input_labels, output_labels, average=None)
        recall_perclass = metrics.recall_score(input_labels, output_labels, average=None)

        classes_by_recall = sorted(classes, reverse=True,
            key=lambda c: (recall_perclass[classes.index(c)], f1_perclass[classes.index(c)]))

        loss_record.append((epoch_loss, assay['eval_loss']))
        if plotutil: plotutil.loss(loss_record, output=savepath+'/loss_plot.png', title='{} Loss'.format(run_name))

        record_epoch_stats(epoch=epoch, eval_dict=assay, append=True, name=run_name,
                           training_loss=epoch_loss,
                           savepath=savepath, filename='evaluation_records.lod', cli_args=cli_args,
                           secs_elapsed=ts2secs(training_startime, time.strftime(YMD_HMS)))

        # Saving the model if it's the best
        if assay['eval_loss'] < best_loss:
            # saving model
            torch.save({'state_dict': model.state_dict(),
                        'classes':    classes,
                        'type':       cli_args['model'],
                        'version':    {'torch':       torch.__version__,
                                       'torchvision': torchvision.__version__}
                        },
                       '{}/model.pt'.format(savepath))

            # saving training results
            record_epoch_stats(epoch=epoch, eval_dict=assay, append=False, name=run_name,
                               training_loss=epoch_loss,
                               savepath=savepath, filename='best_epoch.dict', cli_args=cli_args,
                               secs_elapsed=ts2secs(training_startime, time.strftime(YMD_HMS)))

            title = '{}, f1_weighted={:.2f}% (epoch {})'.format(run_name, 100*f1_weighted, epoch)
            if plotutil:
                plotutil.make_confusion_matrix_plot(input_labels, output_labels, classes_by_recall, title,
                                          output=savepath+'/confusion_matrix.png', text_as_percentage=True)

        # terminal output
        now = time.strftime(YMD_HMS)
        epoch_elapsed = ts2secs(epoch_startime, now)
        total_elapsed = ts2secs(training_startime, now)
        output_string = '\nEpoch {} done in {:.0f} mins. loss={:.3f}. f1_w={:.1f}% f1_m={:.1f}%. Total elapsed time={:.2f} hrs'
        print(output_string.format(epoch, epoch_elapsed/60, assay['eval_loss'],
                                   100*f1_weighted, 100*f1_macro, total_elapsed/(60*60)))

        best_epoch_text = 'Best Epoch was {} with a validation loss of {:.3f} and an F1 score of {:.1f}%'
        # note or record best epoch, or stop early
        if best_loss > assay['eval_loss']:
            best_loss = assay['eval_loss']
            best_epoch = epoch
            best_f1 = f1_weighted
        elif epoch > 1.33*best_epoch and epoch >= min_epochs:
            print("Model probably won't be improving more from here, shutting this operation down")
            print(best_epoch_text.format(best_epoch, best_loss, 100*best_f1))
            return
        if best_epoch != epoch:
            print(best_epoch_text.format(best_epoch, best_loss, 100*best_f1))


def eval_loop(model, eval_loader, device, criterion):
    """ returns dict of: accuracy, loss, perclass_accuracy, perclass_correct, perclass_totals,
                         true_inputs, predicted_outputs, f1_avg, perclass_f1, classes )"""

    classes = eval_loader.dataset.classes

    eval_loss = 0
    all_true_input_labels = []
    all_input_image_paths = []
    all_predicted_output_labels = []
    all_predicted_output_ranks = []
    all_predicted_output_allranks = []  # all the scores for all classes, not just max

    model.eval()
    with torch.no_grad():
        criterion = criterion.to(device)
        for i, data in enumerate(eval_loader):

            # forward pass
            input_images, true_input_labels, input_image_paths = data
            input_images, true_input_labels = input_images.to(device), true_input_labels.to(device)

            # run model and determin loss
            outputs = model(input_images)  # eval-loop
            try:
                outputs = outputs.to(device)
                loss = criterion(outputs, true_input_labels)
            except AttributeError:  # e="'tuple' object has no attribute 'to'" when inception_v3 aux_logits=True
                outputs, aux_outputs = outputs
                outputs, aux_outputs = outputs.to(device), aux_outputs.to(device)
                loss1 = criterion(outputs, true_input_labels)
                loss2 = criterion(aux_outputs, true_input_labels)
                loss = loss1+0.4*loss2

            eval_loss += loss.item()

            # format results
            outputs = F.softmax(outputs, dim=1)
            predicted_output_ranks, predicted_output_labels = torch.max(outputs, 1)
            # outputs format is a tensor list of lists, inside lists are len(classes) long,
            # each with a per-class weight prediction. outide list is batch_size long.
            # doing torch.max returns the index of the highest per-class prediction (inside list),
            # for all elements of outside list. Therefor len(predicted)==len(output)==batch_size

            #all_predicted_output_allranks.extend(outputs.tolist())
            all_predicted_output_labels.extend([p.item() for p in predicted_output_labels])
            all_predicted_output_ranks.extend([r.item() for r in predicted_output_ranks])
            all_true_input_labels.extend([t.item() for t in true_input_labels])
            all_input_image_paths.extend(input_image_paths)

            # printing progress
            testing_progressbar(i, len(eval_loader))

    # result metrics
    assert len(all_predicted_output_labels) == \
           len(all_true_input_labels) == \
           len(all_input_image_paths)
    assay = dict(eval_loss=eval_loss,
                 classes=classes,
                 true_inputs=all_true_input_labels,
                 prediction_outputs=all_predicted_output_labels,
                 prediction_ranks=all_predicted_output_ranks,
                 #prediction_allscores=all_predicted_output_allranks,
                 true_images=all_input_image_paths)

    return assay


## INITIALIZATION ##

import argparse

if __name__ == '__main__':
    model_choices = ['inception_v3', 'alexnet', 'squeezenet',
                     'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet151',
                     'densenet121', 'densenet169', 'densenet161', 'densenet201']
    augmentation_choices = ['flipx', 'flipy', 'flipxy', 'training-only']

    ## Parsing command line input ##
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help='root src directory containing class folders')
    parser.add_argument("output_dir", help="directory to output logs and saved model")
    parser.add_argument("--split", required=True,
                        help='ratio of files to split into training and evaluation datasets. eg: "80:20" ')
    parser.add_argument("--class-config", nargs=2,
                        help='path to a configuration csv + column-header. Used for skipping and grouping classes in SRC')
    parser.add_argument("--seed", type=int,
                        help="specifying a seed value allows reproducability when randomizing images for training and evaluation")
    parser.add_argument("--swap", "--swap-datasets-around", action='store_true',
                        help="swap training and evaluation datasets with each other. Useful for 50:50 dupe runs.")
    parser.add_argument("--class-minimum", type=int, default=2,
                        help='the minimum viable number of images per class. Classes with fewer pre-split instances than this value will not be included.')
    #TODO any if either dataset must drop a class, both datasets drop the class.
    #TODO Can be specified pre-SPLIT (single number) and post-SPLIT (colon-deliminated). eg "10" and "8:2" would be equivalent if SPLIT==80:20,
    #     but 8:3 would also be allowed, forcing eval to be the limiting factor with at least 3 not two instances needed.
    parser.add_argument("--model", default='inception_v3', choices=model_choices,
                        help="select a model architecture to train (default to inceptin_v3)")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help='Preloads model with weights trained on imagenet')
    parser.add_argument("--wn", default=False, action='store_true',
                        help='if included, classes will be weight-normalized during training. This may boost classes with fewer instances.')
    parser.add_argument("--max-epochs", default=60, type=int,
                        help="Maximum Number of Epochs (default 60).\nTraining may end before <max-epochs> if evaluation loss doesn't improve beyond best_epoch*1.33")
    parser.add_argument("--min-epochs", default=16, type=int,
                        help="Minimum number of epochs to run (default 10)")
    parser.add_argument("--batch-size", default=108, type=int, dest='batch_size',
                        help="how many images to process in a batch, (default 108, a number divisible by 1,2,3,4 to ensure even batch distribution across up to 4 GPUs)")
    parser.add_argument("--loaders", default=4, type=int,
                        help='total number of threads to use for loading data to/from GPUs. 4 per GPU is good. (Default 4 total)')
    parser.add_argument("--augment", default=[], nargs='+', choices=augmentation_choices,
                        help='''Data augmentation can improve training. Listed transformations -may- be applied to any given input image when loaded. 
                               "flipx" and "flipy" denotes mirroring the image vertically and horizontally respectively.
                                If "training-only" is included, augmentation transformations will only be applied to the training set.''')
    parser.add_argument("--learning-rate", '--lr', default=0.001, type=float,
                        help='The (initial) learning rate of the training optimizer (Adam). Default is 0.001')

    args = parser.parse_args()  # ingest CLI arguments

    args.output_dir = args.output_dir.rstrip(os.sep)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
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

    ## initializing data ##
    print('Initializing Data...')
    if not args.class_config:
        nd = NeustonDataset(src=args.src, minimum_images_per_class=args.class_minimum)
    else:
        nd = NeustonDataset.from_csv(csv_file=args.class_config[0], column_to_run=args.class_config[1],
                                     src=args.src, minimum_images_per_class=args.class_minimum)
        # TODO record which classes were grouped and skipped.
    ratio1, ratio2 = map(int, args.split.split(':'))

    if args.seed is None:
        args.seed = random.randrange(sys.maxsize)

    dataset_tup = nd.split(ratio1, ratio2, seed=args.seed)
    if args.swap:
        evaluation_dataset, training_dataset = dataset_tup
    else:
        training_dataset, evaluation_dataset = dataset_tup

    ci_nd = nd.classes_ignored_from_too_few_samples
    ci_train = training_dataset.classes_ignored_from_too_few_samples
    ci_eval = evaluation_dataset.classes_ignored_from_too_few_samples
    assert ci_eval == ci_train
    if ci_nd:
        msg = '\n{} out of {} classes ignored from --class-minimum {}, PRE-SPLIT'
        msg = msg.format(len(ci_nd), len(nd.classes+ci_nd), args.class_minimum)
        ci_nd = ['({:2}) {}'.format(l, c) for c, l in ci_nd]
        print('\n    '.join([msg]+ci_nd))
    if ci_eval:
        msg = '\n{} out of {} classes ignored from --class-minimum {}, POST-SPLIT'
        msg = msg.format(len(ci_eval), len(evaluation_dataset.classes+ci_eval), args.class_minimum)
        ci_eval = ['({:2}) {}'.format(l, c) for c, l in ci_eval]
        print('\n    '.join([msg]+ci_eval))
    print()

    ## TESTING SEED ##
    #images = training_dataset.images + evaluation_dataset.images
    #with open(args.output_dir+'/seed_{}_testA.list'.format(args.seed),'w') as f:
    #    f.write('\n'.join(images)+'\n')
    #print('seed file written')
    #import sys; sys.exit()
    ## END TESTING SEED ##

    # Transforms and augmentation #
    pixels = [299, 299] if args.model == 'inception_v3' else [224, 224]
    base_tforms = [transforms.Resize(pixels),
                   transforms.ToTensor()]

    if args.augment != []:
        # when data is loaded into model, there is a 50% chance per axis it will be transformed
        aug_tforms = []
        if any([arg in args.augment for arg in ['flip', 'flipxy', 'flip-xy', 'flip-both']]):
            aug_tforms.append(transforms.RandomVerticalFlip(p=0.5))
            aug_tforms.append(transforms.RandomHorizontalFlip(p=0.5))
        elif any([arg in args.augment for arg in ['flipx', 'flip-x', 'flip-vertical']]):
            aug_tforms.append(transforms.RandomVerticalFlip(p=0.5))
        elif any([arg in args.augment for arg in ['flipy', 'flip-y', 'flip-horizontal']]):
            aug_tforms.append(transforms.RandomHorizontalFlip(p=0.5))

        train_tforms = transforms.Compose(aug_tforms+base_tforms)
        if 'training-only' in args.augment:
            eval_tforms = transforms.Compose(base_tforms)
        else:
            eval_tforms = train_tforms
    # no augmentation #
    else:
        args.augment = False
        train_tforms = eval_tforms = transforms.Compose(base_tforms)

    # applying transforms
    training_dataset.transforms = train_tforms
    evaluation_dataset.transforms = eval_tforms

    # create dataloaders
    print('Loading Training Dataloader...')
    training_loader = torch.utils.data.DataLoader(training_dataset, pin_memory=True, shuffle=True,
                                                  batch_size=args.batch_size, num_workers=args.loaders)
    print('Loading Evaluation Dataloader...')
    evaluation_loader = torch.utils.data.DataLoader(evaluation_dataset, pin_memory=True, shuffle=False,
                                                    batch_size=args.batch_size, num_workers=args.loaders)

    # number of classes
    assert training_loader.dataset.classes == evaluation_loader.dataset.classes
    num_o_classes = len(training_loader.dataset.classes)

    ## initializing model  ##
    # options:
    #     inception_v3, alexnet, squeezenet,
    #     vgg11, vgg13, vgg16, vgg19,
    #     vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
    #     resnet18, resnet34, resnet50, resnet101, resnet151,
    #     densenet121, densenet169, densenet161, densenet201
    print('Adjusting model output layer for {} classes...'.format(num_o_classes))
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
    else:
        raise KeyError("model unknown!")

    # multiple GPU wrapper
    if torch.cuda.device_count() > 1:  # if multiple-gpu's detected, use available gpu's
        model = nn.DataParallel(model, device_ids=list(range(len(gpus))))
    model.to(device)  # sends model to GPU

    # optimizerx & loss function criterion
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss
    if args.wn: # weight normalization
        # "nd" is the unsplit original NeustonDataset
        perclass_count = {class_label: len(imgs) for class_label, imgs in nd.images_perclass.items()}
        const = len(nd.imgs)/len(perclass_count)
        weights = [const/count for count in perclass_count.values()]
        print('\nCLASS: WEIGHT [count]')
        print('Weight = ( total_image_count / number_of_classes ) /count_perclass')
        print('       = ( {} / {} )/count = ~{:.0f}/count'.format(len(training_loader.dataset.imgs), len(perclass_count), const))
        [print('{:>30}: {: 5.2f} [{}]'.format(c, w, n)) for c, n, w in
         zip(perclass_count.keys(), perclass_count.values(), weights)]
        print()
        weights = torch.FloatTensor(weights).to(device)
    else:
        weights = None

    try:
        criterion = criterion(weight=weights).to(device)
    except TypeError:
        criterion = criterion().to(device)

    print("Model: {}".format(model.__class__))
    print("Transformations: {}".format(train_tforms))
    print("Epochs: {}-{}, Batch size: {}".format(args.min_epochs, args.max_epochs, args.batch_size))

    training_loop(model, training_loader, evaluation_loader, device, args.output_dir, optimizer, criterion,
                  weights=weights, max_epochs=args.max_epochs, min_epochs=args.min_epochs, cli_args=vars(args))


