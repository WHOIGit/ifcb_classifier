#! /usr/bin/env python

## Builtin Imports ##
import os
import time
import fnmatch
from ast import literal_eval

## 3rd party library imports ##
from sklearn import metrics

## TORCH STUFF ##
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import dataset
import torchvision.models as MODEL_MODULE

## local imports ##
try: import plotutil  # installation of matplotlib is optional
except ImportError: plotutil=None

YMD_HMS = "%Y-%m-%d %H:%M:%S"

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
        return data, target,path


def record_epoch_stats(epoch, eval_dict, training_loss, cli_args, name, savepath, filename, secs_elapsed, append=False):
    save_dict = eval_dict.copy()
    save_dict['name'] = name
    save_dict['epoch'] = epoch
    save_dict['secs_elapsed'] = secs_elapsed
    save_dict['training_loss'] = training_loss
    save_dict.update(cli_args)

    fpath = '{}/{}'.format(savepath, filename)
    #print('writing to', fpath, '... ', end='')

    if append:
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
                  normalize_weights=False, max_epochs=100, min_epochs=10, epoch0=0, best_loss=float('inf'), cli_args={}):
    run_name = os.path.basename(savepath)
    num_o_batches = len(training_loader)
    best_epoch = epoch0
    best_f1 = 0
    loss_record = []  # with one train-eval loss tuple per epoch

    print('Training... there are {} batches per epoch'.format(num_o_batches))
    print('Starting at epoch {} of max {}'.format(epoch0+1, max_epochs))

    ## defining loss function criterion
    try: # dataset concat clause
        dataset = training_loader.dataset.datasets[0]
    except AttributeError:
        # regular
        dataset = training_loader.dataset

    if normalize_weights:
        weights = [0 for c in dataset.classes]
        for img, img_idx in dataset.imgs:
            weights[img_idx] += 1
        weights = [len(dataset.imgs)/len(weights)/weight for weight in weights]
        weights = torch.FloatTensor(weights).to(device)
    else:
        weights = None

    try:
        criterion = criterion(weight=weights).to(device)
    except TypeError:
        criterion = criterion.to(device)

    # start training loop
    training_startime = time.strftime(YMD_HMS)
    for epoch in range(epoch0+1, max_epochs+1):
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

            # run model and determine loss
            outputs = model(inputs) #training-loop
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
        input_labels =   [classes[c] for c in assay['true_inputs']]
        output_labels =  [classes[c] for c in assay['prediction_outputs']]
        f1_weighted =     metrics.f1_score(assay['true_inputs'], assay['prediction_outputs'], average='weighted')
        f1_macro =        metrics.f1_score(assay['true_inputs'], assay['prediction_outputs'], average='macro')
        f1_perclass =     metrics.f1_score(      input_labels,   output_labels, average=None)
        recall_perclass = metrics.recall_score(  input_labels,   output_labels, average=None)

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
            torch.save(model.state_dict(), '{}/model.pt'.format(savepath))

            # saving training results
            record_epoch_stats(epoch=epoch, eval_dict=assay, append=False, name=run_name,
                               training_loss=epoch_loss,
                               savepath=savepath, filename='best_epoch.dict', cli_args=cli_args,
                               secs_elapsed=ts2secs(training_startime, time.strftime(YMD_HMS)))

            title = '{}, f1_weighted={:.2f}% (epoch {})'.format(run_name, 100*f1_weighted, epoch)
            if plotutil: plotutil.make_confusion_matrix_plot(input_labels, output_labels, classes_by_recall, title,
                                  outfile=savepath+'/confusion_matrix.png', show=False, text_as_percentage=True)


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


def eval_loop(model, eval_loader, device, criterion=nn.CrossEntropyLoss()):
    """ returns dict of: accuracy, loss, perclass_accuracy, perclass_correct, perclass_totals,
                         true_inputs, predicted_outputs, f1_avg, perclass_f1, classes )"""
    try:
        classes = eval_loader.dataset.classes
    except AttributeError:
        classes = eval_loader.dataset.datasets[0].classes

    eval_loss = 0
    all_true_input_labels = []
    all_input_image_paths = []
    all_predicted_output_labels = []
    all_predicted_output_ranks  = []

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
            testing_progressbar(i, len(eval_loader))

    # result metrics
    assert len(all_predicted_output_labels) == \
           len(all_true_input_labels) == \
           len(all_input_image_paths)
    assay = dict(eval_loss=eval_loss,
                 classes=classes,
                 true_inputs=all_true_input_labels,
                 prediction_outputs=all_predicted_output_labels,
                 #prediction_ranks=all_predicted_output_ranks,
                 true_images=all_input_image_paths)

    return assay


## INITIALIZATION ##

import argparse

if __name__ == '__main__':
    model_choices = ['inception_v3', 'alexnet', 'squeezenet',
                     'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet151',
                     'densenet121', 'densenet169', 'densenet161', 'densenet201']
    augmentation_choices = ['flipx', 'flipy', 'flipxy', 'training-only', 'nochance']

    ## Parsing command line input ##
    parser = argparse.ArgumentParser()
    parser.add_argument("training_dir", help="path to training set images")
    parser.add_argument("evaluation_dir", help="path to testing set images")
    parser.add_argument("output_dir", help="directory out output logs and saved model")
    parser.add_argument("--model", default='inception_v3', choices=model_choices,
                        help="select a model architecture to train (default to inceptin_v3)")
    parser.add_argument("--pretrained", default=True, action='store_true',
                        help='Preloads model with weights trained on imagenet')
    parser.add_argument("--no-normalize", default=False, action='store_true',
                        help='if included, classes will NOT will be weighted during training. Classes with fewer instances will not train as well')
    parser.add_argument("--max-epochs", default=100, type=int,
                        help="Maximum Number of Epochs (default 100).\nTraining may end before <max-epochs> if evaluation loss doesn't improve beyond best_epoch*1.33")
    parser.add_argument("--min-epochs", default=10, type=int,
                        help="Minimum number of epochs to run (default 10)")
    parser.add_argument("--batch-size", default=108, type=int, dest='batch_size',
                        help="how many images to process in a batch, (default 108, a number divisible by 1,2,3,4 to ensure even batch distribution across up to 4 GPUs)")
    parser.add_argument("--loaders", default=4, type=int,
                        help='total number of threads to use for loading data to/from GPUs. 4 per GPU is good. (Default 4 total)')

    parser.add_argument("--augment", default=[], nargs='+', choices=augmentation_choices, help='''Data augmentation can improve training. Listed transformations -may- be applied to any given input image when loaded. 
        To deterministically apply transformations to all input images while also retaining non-transformed images, use "nochance".
        flipx and flipy denotes mirroring the image vertically and horizontally respectively.
        If "training-only" is included, augmentation transformations will only be applied to the training set.''')
    parser.add_argument("--learning-rate", '--lr', default=0.001, type=float,
                        help='The (initial) learning rate of the training optimizer (Adam). Default is 0.001')

    args = parser.parse_args()  # ingest CLI arguments

    # input cleanup
    if not args.evaluation_dir.startswith('data/'):
        args.evaluation_dir = 'data/'+args.evaluation_dir
    if not os.path.exists(args.evaluation_dir):
        os.makedirs(args.evaluation_dir)
    if not args.training_dir.startswith('data/'):
        args.training_dir = 'data/'+args.training_dir

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

    # setting model appropriate resize
    if args.model == 'inception_v3':
        pixels = [299, 299]
    else:
        pixels = [224, 224]

    ## initializing data ##
    base_tforms = [transforms.Resize(pixels),  # 299x299 is minimum size for inception
                   transforms.ToTensor()]


    training_dataset_func = datasets.ImageFolder
    evaluation_dataset_func = ImageFolderWithPaths

    # augmentation #
    if 'nochance' in args.augment:
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
            training_datasets.append(training_dataset_func(root=args.training_dir, transform=tforms))
            validation_datasets.append(evaluation_dataset_func(root=args.evaluation_dir, transform=tforms))
        training_dataset = torch.utils.data.ConcatDataset(training_datasets)
        if 'training-set-only' in args.augment:
            tforms = transforms.Compose(base_tforms)
            evaluation_dataset = evaluation_dataset_func(root=args.evaluation_dir, transform=tforms)
        else:
            evaluation_dataset = torch.utils.data.ConcatDataset(validation_datasets)
        tforms = tforms_lol  # for cli printint later

    elif args.augment != []:
        # when data is loaded into model, there is a 50% chance per axis it will be transformed
        aug_tforms = []
        if any([arg in args.augment for arg in ['flip', 'flipxy', 'flip-xy', 'flip-both']]):
            aug_tforms.append(transforms.RandomVerticalFlip(p=0.5))
            aug_tforms.append(transforms.RandomHorizontalFlip(p=0.5))
        elif any([arg in args.augment for arg in ['flipx', 'flip-x', 'flip-vertical']]):
            aug_tforms.append(transforms.RandomVerticalFlip(p=0.5))
        elif any([arg in args.augment for arg in ['flipy', 'flip-y', 'flip-horizontal']]):
            aug_tforms.append(transforms.RandomHorizontalFlip(p=0.5))
        if 'random-crop-598' in args.augment:
            aug_tforms.append(transforms.RandomCrop(589))
        tforms = transforms.Compose(aug_tforms+base_tforms)
        training_dataset = training_dataset_func(root=args.training_dir, transform=tforms)
        if 'training-set-only' in args.augment:
            tforms = transforms.Compose(base_tforms)
        evaluation_dataset = evaluation_dataset_func(root=args.evaluation_dir, transform=tforms)

    # no augmentation #
    else:
        args.augment = False
        tforms = transforms.Compose(base_tforms)
        training_dataset = training_dataset_func(root=args.training_dir, transform=tforms)
        evaluation_dataset = evaluation_dataset_func(root=args.evaluation_dir, transform=tforms)

    # create dataloaders
    print('Loading Training Data...')
    training_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True, batch_size=args.batch_size,
                                                  pin_memory=True, num_workers=args.loaders)
    print('Loading Validation Data...')
    evaluation_loader = torch.utils.data.DataLoader(evaluation_dataset, shuffle=True, batch_size=args.batch_size,
                                                    pin_memory=True, num_workers=args.loaders)

    # number of classes
    try:
        assert training_loader.dataset.classes == evaluation_loader.dataset.classes
        num_o_classes = len(training_loader.dataset.classes)
    except AttributeError:
        # dataset concat clause
        assert training_loader.dataset.datasets[0].classes == evaluation_loader.dataset.datasets[0].classes
        num_o_classes = len(training_loader.dataset.datasets[0].classes)
    print("Number of Classes:", num_o_classes)


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

    # multiple GPU wrapper
    if torch.cuda.device_count() > 1:  # if multiple-gpu's detected, use available gpu's
        model = nn.DataParallel(model, device_ids=list(range(len(gpus))))
    model.to(device)  # sends model to GPU

    # optimizer
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate )
    criterion = nn.CrossEntropyLoss

    print("Model: {}".format(model.__class__))
    print("Transformations: {}".format(tforms))
    print("Epochs: {}-{}, Batch size: {}".format(args.min_epochs,args.max_epochs, args.batch_size))

    training_loop(model, training_loader, evaluation_loader, device, args.output_dir, optimizer, criterion,
                  normalize_weights=args.no_normalize, max_epochs=args.max_epochs, min_epochs=args.min_epochs, cli_args=vars(args))


