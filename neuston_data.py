"""this module handles the parsing of data directories"""

# built in imports
import os, sys
import random

# 3rd party imports
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset, IterableDataset
from torch import Tensor
import pandas as pd

# project imports
import ifcb
from ifcb.data.adc import SCHEMA_VERSION_1
from ifcb.data.stitching import InfilledImages


## TRAINING ##

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
            if d1_len == len(images) and self.minimum_images_per_class>1:
            # make sure that at least one image gets put in d2
                d1_len -= 1

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
            d1_threshold = int(self.minimum_images_per_class*ratio1/100+0.5) or 1
            d2_threshold = self.minimum_images_per_class-d1_threshold or 1
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


def get_trainval_datasets(args):
    ## initializing data ##
    print('Initializing Data...')
    if not args.class_config:
        nd = NeustonDataset(src=args.SRC, minimum_images_per_class=args.class_min)
    else:
        nd = NeustonDataset.from_csv(csv_file=args.class_config[0], column_to_run=args.class_config[1],
                                     src=args.SRC, minimum_images_per_class=args.class_min)
        # TODO record which classes were grouped and skipped.
    ratio1, ratio2 = map(int, args.split.split(':'))

    dataset_tup = nd.split(ratio1, ratio2, seed=args.seed)
    if False: # srgs.swap: #TODO: DUPES
        validation_dataset, training_dataset = dataset_tup
    else:
        training_dataset, validation_dataset = dataset_tup

    ci_nd = nd.classes_ignored_from_too_few_samples
    ci_train = training_dataset.classes_ignored_from_too_few_samples
    ci_eval = validation_dataset.classes_ignored_from_too_few_samples
    assert ci_eval == ci_train
    if ci_nd:
        msg = '\n{} out of {} classes ignored from --class-minimum {}, PRE-SPLIT'
        msg = msg.format(len(ci_nd), len(nd.classes+ci_nd), args.class_min)
        ci_nd = ['({:2}) {}'.format(l, c) for c, l in ci_nd]
        print('\n    '.join([msg]+ci_nd))
    if ci_eval:
        msg = '\n{} out of {} classes ignored from --class-minimum {}, POST-SPLIT'
        msg = msg.format(len(ci_eval), len(validation_dataset.classes+ci_eval), args.class_min)
        ci_eval = ['({:2}) {}'.format(l, c) for c, l in ci_eval]
        print('\n    '.join([msg]+ci_eval))

    # applying transforms
    train_tforms, val_tforms = get_trainval_transforms(args)
    training_dataset.transforms = train_tforms
    validation_dataset.transforms = val_tforms

    return training_dataset, validation_dataset


## transforms and augmentation ##
def get_trainval_transforms(args):
    # Transforms  #
    args.resize = 299 if args.MODEL == 'inception_v3' else 224
    tform_resize = transforms.Resize([args.resize,args.resize])
    tform_img_norm = transforms.Normalize(3*[args.img_norm[0]],3*[args.img_norm[1]]) if args.img_norm else None
    base_tforms = [tform_resize, transforms.ToTensor()]
    if tform_img_norm: base_tforms.append(tform_img_norm)

    aug_tforms_training = []
    aug_tforms_validation = []
    if args.flip:
        flip_tforms = []
        # args.flip choices=[x y xy x+V y+V xy+V]
        if 'x' in args.flip:
            flip_tforms.append(transforms.RandomVerticalFlip(p=0.5))
        if 'y' in args.flip:
            flip_tforms.append(transforms.RandomHorizontalFlip(p=0.5))

        aug_tforms_training.extend(flip_tforms)
        if '+V' in args.flip: aug_tforms_validation.extend(flip_tforms)

    # TODO add other augments here

    train_tforms = transforms.Compose( aug_tforms_training + base_tforms )
    val_tforms = transforms.Compose( aug_tforms_validation + base_tforms )

    return train_tforms, val_tforms

## RUNNING ##

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

# untested
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
                img = Tensor([img]*3)
                img = transforms.Resize(self.resize)(transforms.ToPILImage()(img))
                img = transforms.ToTensor()(img)
                yield img, target_pid

    def __len__(self):
        """warning: for large datasets, this is very very slow"""
        return sum(len(bin) for bin in self.dd)


class IfcbBinDataset(Dataset):
    def __init__(self, bin, resize):
        self.bin = bin
        self.images = []
        self.pids = []

        # use 299x299 for inception_v3, all other models use 244x244
        if isinstance(resize, int):
            resize = (resize, resize)
        self.resize = resize

        # old-style bins need to be stitched and infilled
        if bin.schema == SCHEMA_VERSION_1:
            bin_images = InfilledImages(bin)
        else:
            bin_images = bin.images

        for target_number, img in bin_images.items():
            target_pid = bin.pid.with_target(target_number)
            self.images.append(img)
            self.pids.append(target_pid)

    def __getitem__(self, item):
        img = self.images[item]
        img = transforms.ToPILImage(mode='L')(img)
        img = img.convert('RGB')
        img = transforms.Resize(self.resize)(img)
        img = transforms.ToTensor()(img)
        return img, self.pids[item]

    def __len__(self):
        return len(self.pids)

def get_run_dataset():
    pass
