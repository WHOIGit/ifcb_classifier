#! /usr/bin/env python
""" get a satiation prediction from 0-1 using a neuston_net model """

# builtin imports
import argparse
import fnmatch, glob
import os

# Torch imports
import torch
import torch.nn as nn
import torchvision.models as MODEL_MODULE
from torchvision.datasets.folder import default_loader
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader



def load_model(model_type, model_path, device:str='cpu'):
    # assumes model is trained on gpu

    output_layer_size = 105  # due to linear regression
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


    if device == torch.device('cpu'):
        model.load_state_dict( torch.load(model_path, map_location=device) )
    else: # torch.device('cuda')
        model.load_state_dict(torch.load(model_path))
        model.to( device )

    model.eval()
    return model


def run_model(model, input_loader, device):

    all_input_image_paths = []
    all_predicted_output_classes = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(input_loader):

            # forward pass
            input_images, input_image_paths = data
            input_images = input_images.to(device)

            # run model and determin loss
            outputs = model(input_images)
            if isinstance(outputs,tuple):  # e="'tuple' object has no attribute 'to'" when inception_v3 aux_logits=True
                outputs, aux_outputs = outputs

            # format results
            output_ranks, output_classes = torch.max(outputs, 1)
            # outputs format is a tensor list of lists, inside lists are len(classes) long,
            # each with a per-class weight prediction. the outide list is batch_size long.
            # doing torch.max returns the index of the highest per-class prediction (inside list),
            # for all elements of outside list. Therefor len(predicted)==len(output)==batch_size
            # output_classes are a list of class/node indexes, not string labels

            all_predicted_output_classes.extend([p.item() for p in output_classes])
            all_input_image_paths.extend(input_image_paths)

            # printing progress
            testing_progressbar(i, len(input_loader))

    # result metrics
    assay = dict(inputs=all_input_image_paths,
                 outputs=all_predicted_output_classes)
    print()
    return assay

def testing_progressbar(batch: int, batches: int, batches_per_dot: int = 1, batches_per_line: int = 100):
    if batch%batches_per_line == batches_per_line-1 or batch == 0:
        phase_percent_done = 100*batch/batches
        print('\nVerification {:1.1f}% complete'.format(phase_percent_done), end=' ', flush=True)

    if batch%batches_per_dot == batches_per_dot-1:
        print('.', end='', flush=True)


class ImageDataset(Dataset):
    """
    Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    Example setup:      dataloader = torch.utils.DataLoader(ImageFolderWithPaths("path/to/your/perclass/image/folders"))
    Example usage:     for inputs,labels,paths in my_dataloader: ....
    instead of:        for inputs,labels in my_dataloader: ....
    adapted from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, image_paths, resize=244):
        self.image_paths = [img for img in image_paths \
                            if any([img.endswith(ext) for ext in datasets.folder.IMG_EXTENSIONS])]

        # use 299x299 for inception_v3, all other models use 244x244
        self.transform =  transforms.Compose([transforms.Resize([resize,resize]),
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

if __name__ == '__main__':
    print("pyTorch VERSION:", torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to the trained ifcb classifier model')
    parser.add_argument('src', nargs='+', help='one or more image paths or directories')
    parser.add_argument('--filter', default='*.png', help='keep from src only files that match --filter, default is *.png')
    parser.add_argument("--batch-size", default=108, type=int, dest='batch_size',
                        help="how many images to process in a batch, (default 108, a number divisible by 1,2,3,4 to ensure even batch distribution across up to 4 GPUs)")
    parser.add_argument("--loaders", default=4, type=int,
                        help='total number of threads to use for loading data to/from GPUs. 4 per GPU is good. (Default 4 total)')
    parser.add_argument("--stdout", default=False, action='store_true', help="outputs results to stdout")
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
    model = load_model('inception_v3',args.model, device)

    if torch.cuda.device_count() > 1:  # if multiple-gpu's detected, use available gpu's
        model = nn.DataParallel(model, device_ids=list(range(len(gpus))))

    ## ingesting input
    inputs = []
    for elem in args.src:
        if os.path.isfile(elem) and fnmatch.fnmatch(elem,args.filter):
            inputs.append(elem)
        elif os.path.isdir(elem):
            matches = glob.glob(os.path.join(elem,'**',args.filter), recursive=True)
            matches.sort()
            inputs.extend(matches)
        else:
            print(elem,'is not a file or a directory')
            # todo injest a file list

    print('Files to ingest:', len(inputs))
    image_dataset = ImageDataset(inputs, resize=299)
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size,
                              pin_memory=True, num_workers=args.loaders)

    results = run_model(model, image_loader, device)
    # returns {inputs:[...], outputs:[...]}
    print()
    if args.stdout:
        for i,o in zip(*results.values()):
            print(o,i)



