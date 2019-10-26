#! /usr/bin/env python
""" get a satiation prediction from 0-1 using a neuston_net model """

# builtin imports
import argparse
import fnmatch, glob
import os, ast
import h5py as h5

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as MODEL_MODULE
from torchvision.datasets.folder import default_loader
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

# 3rd Party Imports
import ifcb
import numpy as np


def save_class_scores_hdf(path, bin_id, scores, roi_ids, class_labels):
    np_scores = np.array(scores)
    assert np_scores.shape[0] == len(roi_ids), 'wrong number of ROI numbers'
    assert np_scores.shape[1] == len(class_labels), 'wrong number of class labels'

    try:
        with h5.File(path,'w') as f:
            ds = f.create_dataset('scores', data=np_scores)
            ds.attrs['bin_id'] = bin_id
            ds.attrs['class_labels'] = [l.encode('ascii') for l in class_labels]
            ds.attrs['roi_numbers'] = [ifcb.Pid(roi_id).target for roi_id in roi_ids]
    except RuntimeError as e:
        # RuntimeError: Unable to create attribute (object header message is too large)
        # see: https://github.com/h5py/h5py/issues/1053#issuecomment-525363860
        print('   ',type(e),e, 'SWITCHING TO CSV OUTPUT')
        path = path.replace('.h5','.csv').replace('.hdf','.csv')
        save_class_scores_csv(path, bin_id, scores, roi_ids, class_labels)


def save_class_scores_csv(path, bin_id, scores, roi_ids, classes):
    roi_nums = [ifcb.Pid(roi_id).target for roi_id in roi_ids]
    with open(path, 'w') as f:
        header = bin_id+',Highest_Ranking_Class,'+','.join(classes)
        f.write(header+os.linesep)
        for image_id, class_ranks_by_id in zip(roi_nums,scores):
            top_class = class_ranks_by_id.index(max(class_ranks_by_id))
            line = '{},{}'.format(image_id, classes[top_class])
            line_ranks = ','.join([str(r) for r in class_ranks_by_id])
            line = '{},{}'.format(line, line_ranks)
            f.write(line+os.linesep)

def load_model(model_path, device=torch.device('cpu')):
    # assumes model is trained on gpu

    if device == torch.device('cpu'):
        model_dict = torch.load(model_path, map_location=device)
    else:  # torch.device('cuda')
        model_dict = torch.load(model_path)

    model_type = model_dict['type']
    state_dict = model_dict['state_dict']
    classes = model_dict['classes']
    output_layer_size = len(classes)

    if model_type == 'inception_v3':
        model = MODEL_MODULE.inception_v3()  #, num_classes=num_o_classes, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, output_layer_size)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, output_layer_size)
    elif model_type == 'alexnet':
        model = getattr(MODEL_MODULE, model_type)()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, output_layer_size)
    elif model_type == 'squeezenet':
        model = getattr(MODEL_MODULE, model_type+'1_1')()
        model.classifier[1] = nn.Conv2d(512, output_layer_size, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = output_layer_size
    elif model_type.startswith('vgg'):
        model = getattr(MODEL_MODULE, model_type)()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, output_layer_size)
    elif model_type.startswith('resnet'):
        model = getattr(MODEL_MODULE, model_type)()
        model.fc = nn.Linear(model.fc.in_features, output_layer_size)
    elif model_type.startswith('densenet'):
        model = getattr(MODEL_MODULE, model_type)()
        model.classifier = nn.Linear(model.classifier.in_features, output_layer_size)

    model.load_state_dict(state_dict)
    if device == torch.device("cuda:0"): model.to(device)
    model.eval()
    return model,classes


def run_model(model, input_loader, device):
    all_input_image_IDs = []
    all_predicted_output_classes = []
    all_class_ranks = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(input_loader):

            # forward pass
            input_images, input_image_IDs = data
            input_images = input_images.to(device)

            # run model and determin loss
            outputs = model(input_images)
            if isinstance(outputs, tuple):  #inception_v3 clause
                outputs, aux_outputs = outputs

            # format results
            output_ranks, output_classes = torch.max(outputs, 1)
            # outputs format is a tensor list of lists, inside lists are len(classes) long,
            # each with a per-class weight prediction. the outide list is batch_size long.
            # doing torch.max returns the index of the highest per-class prediction (inside list),
            # for all elements of outside list. Therefor len(predicted)==len(output)==batch_size
            # output_classes are a list of class/node indexes, not string labels

            #TODO capture/log outputs totally
            # per-line output format = image-name,top-class,class1-rank,class2-rank,...classx-rank
            output_classes = [p.item() for p in output_classes]
            all_predicted_output_classes.extend(output_classes)
            all_input_image_IDs.extend(input_image_IDs)
            all_class_ranks.extend(outputs.tolist()) #UNTESTED

            # printing progress
            testing_progressbar(i, len(input_loader))

    # result metrics
    # TODO save model metadata next to model
    # include: model_type, classes by index, torch tensor size format, output-layer size.
    assay = dict(inputs=all_input_image_IDs,
                 outputs=all_predicted_output_classes,
                 outputs_allranks=all_class_ranks)
    return assay


def testing_progressbar(batch: int, batches: int, batches_per_dot: int = 1, batches_per_line: int = 100):
    if batch%batches_per_line == batches_per_line:  # or batch == 0:
        phase_percent_done = 100*batch/batches
        print('\nVerification {:1.1f}% complete'.format(phase_percent_done), end=' ', flush=True)

    if batch%batches_per_dot == batches_per_dot-1:
        print('.', end='', flush=True)


class ImageDataset(Dataset):
    """
    Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    Example setup:     dataloader = torch.utils.DataLoader(ImageFolderWithPaths("path/to/your/perclass/image/folders"))
    Example usage:     for inputs,labels,paths in my_dataloader: ....
    instead of:        for inputs,labels in my_dataloader: ....
    adapted from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, image_paths, resize=244):
        self.image_paths = [img for img in image_paths \
                            if any([img.endswith(ext) for ext in datasets.folder.IMG_EXTENSIONS])]

        # use 299x299 for inception_v3, all other models use 244x244
        self.transform = transforms.Compose([transforms.Resize([resize, resize]),
                                             transforms.ToTensor()])

        if len(self.image_paths) < len(image_paths):
            print('{} non-image files were ommited'.format(len(image_paths)-len(self.image_paths)))
        if len(self.image_paths) == 0:
            raise RuntimeError('No images Loaded!!')

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = datasets.folder.default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, path

    def __len__(self):
        return len(self.image_paths)


try:
    from torch.utils.data import IterableDataset


    class IfcbImageDataset(IterableDataset):
        def __init__(self, data_path, resize):
            self.dd = ifcb.DataDirectory(data_path)

            # use 299x299 for inception_v3, all other models use 244x244
            if isinstance(resize, int):
                resize = (resize, resize)
            self.resize = resize

        def __iter__(self):
            for bin in self.dd:
                print(bin)
                for target_number, img in bin.images.items():
                    target_pid = bin.pid.with_target(target_number)
                    img = torch.Tensor([img]*3)
                    img = transforms.Resize(self.resize)(transforms.ToPILImage()(img))
                    img = transforms.ToTensor()(img)
                    yield img, target_pid

        def __len__(self):
            """warning: for large datasets, this is very very slow"""
            return sum(len(bin) for bin in self.dd)
except ImportError:
    IfcbImageDataset = None


class IfcbBinDataset(Dataset):
    def __init__(self, bin, resize):
        self.bin = bin
        self.images = []
        self.pids = []

        # use 299x299 for inception_v3, all other models use 244x244
        if isinstance(resize, int):
            resize = (resize, resize)
        self.resize = resize

        for target_number, img in bin.images.items():
            target_pid = bin.pid.with_target(target_number)
            self.images.append(img)
            self.pids.append(target_pid)

    def __getitem__(self, item):
        img = self.images[item]
        img = torch.Tensor([img]*3)
        img = transforms.Resize(self.resize)(transforms.ToPILImage()(img))
        img = transforms.ToTensor()(img)
        return img, self.pids[item]

    def __len__(self):
        return len(self.pids)


if __name__ == '__main__':
    print("pyTorch VERSION:", torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('src', nargs='+', help='single file or directory featuring one or more input images or bins. Searches directories recursively.')
    parser.add_argument('--model', required=True, help='path to the trained ifcb classifier model')
    parser.add_argument('--input_type', default='bins', choices=['bins','png'],
                        help='Parse *.png images or *.bin bins. Default is "bins"')
    parser.add_argument("--outdir",  default='.', help='directory results will be saved to')
    parser.add_argument("--outfile", default="{bin}_class_V2.h5", help=
        'If "{bin}" is in --outfile, results are written to files on a per-bin basis where "{bin}" is replaced with the Bin ID.'
        'If --outfile is "stdout", results are output to the terminal in csv format.'
        'If --outfile ends with .h5 or .hdf, file(s) will be hdf formatted, otherwise a csv is created.')

#    parser.add_argument("--noranks", default=False, action='store_true',
#                        help="if included, per-class ranks will not be featured in the output file")
    parser.add_argument("--batch-size", default=108, type=int, dest='batch_size',
                        help="how many images to process in a batch, (default 108, a number divisible by 1,2,3,4 to ensure even batch distribution across up to 4 GPUs)")
    parser.add_argument("--loaders", default=4, type=int,
                        help='total number of threads to use for loading data to/from GPUs. 4 per GPU is good. (Default 4 total)')

    args = parser.parse_args()

    # torch gpu  setup
    if torch.cuda.is_available():
        gpus = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        # pytorch keys off the ~index~ of CUDA_VISIBLE_DEVICES.
        # So if gpus == [3,4] then device "cuda:0" == GPU no. 3
        #                      and device "cuda:1" == GPU no. 4
        device = torch.device('cuda:0')
    else:
        gpus = []
        device = torch.device("cpu")
    print("CUDA_VISIBLE_DEVICES: {}".format(gpus))

    ## loading model
    model,classes = load_model(args.model, device)

    if torch.cuda.device_count() > 1:
        # if multiple-gpu's detected, use all available gpu's
        model = nn.DataParallel(model, device_ids=list(range(len(gpus))))

    resize = 299 if 'inception' in str(model.__class__) else 244

    ## creating output directory
    os.makedirs(args.outdir, exist_ok=True)

    ## ingesting input
    if len(args.src)==1 and os.path.isdir(args.src[0]):
        if args.input_type == 'bins':
            assert '{bin}' in args.outfile
            dd = ifcb.DataDirectory(args.src[0])
            num_of_bins = len(dd)
            for i, bin in enumerate(dd):
                bin_id = os.path.basename(str(bin))
                bin_dataset = IfcbBinDataset(bin, resize)
                image_loader = DataLoader(bin_dataset, batch_size=args.batch_size,
                                          pin_memory=True, num_workers=args.loaders)

                print('{:.02f}% {} images:{}, batches:{}'.format(100*i/num_of_bins, bin, len(bin_dataset),
                                                                 len(image_loader)), end=' -- ', flush=True)

                ## Running the Model ##
                results = run_model(model, image_loader, device)

                print()
                outfile = os.path.join(args.outdir,args.outfile.format(bin=bin_id))
                if args.outfile.endswith('.h5') or args.outfile.endswith('.hdf'):
                    # desired D20180122T180157_IFCB010_class_v2.h5   ie {bin}_class_V2.h5
                    save_class_scores_hdf(outfile, bin_id, results['outputs_allranks'], results['inputs'], classes)
                elif args.outfile.endswith('.csv'):
                    save_class_scores_csv(outfile, bin_id, results['outputs_allranks'], results['inputs'], classes)
                else:
                    raise ValueError
        else: # directory of images
            raise NotImplementedError
    else: # list of images
        raise NotImplementedError

print('Thank you Goodbye!')



    #==============================================================
    #==============================================================


        #TODO too: classic image-based ingestion #
"""     
        inputs = []
        for elem in args.src:
            if os.path.isfile(elem) and fnmatch.fnmatch(elem, args.filter):
                inputs.append(elem)
            elif os.path.isdir(elem):
                matches = glob.glob(os.path.join(elem, '**', args.filter), recursive=True)
                matches.sort()
                inputs.extend(matches)
            else:
                print(elem, 'is not a file or a directory')
                # todo injest a file list

        print('Files to ingest:', len(inputs))
        image_dataset = ImageDataset(inputs, resize)
        image_loader = DataLoader(image_dataset, batch_size=args.batch_size,
                                  pin_memory=True, num_workers=args.loaders)

        results = run_model(model, image_loader, device)
        # returns {inputs:[...], outputs:[...]}
"""



