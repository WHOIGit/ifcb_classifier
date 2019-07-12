# IFCB Classifier

This repo host an image classifying program designed to be trained on plankton images from an IFCB datasource.


# OUTLINE

**data_recomb.py** - a utility tool to copy an whole data set into training and evaluation datasets

**neuston_net.py** - Image classification neural net model trainer

**neuston_run.py** - Processes images using model produced by neuson_net

**plotutil.py** - module to assist with in-training plotting functions

**classif_output_structs.py** - module to help parse, process, plot the output of iterative neuston_net results

**data_reviewing.ipynb** - Jupyter Notebook harnessing classif_output_structs to produce result analysis

**transcribe_dupes.ipynb - Notebook for copying chronicaly misclassified images to a new directory structure for reviewing.
 

# INSTALLATION

Requirement .req files were made with conda.

ifcb.req contains everything needed to run neuston_net and neuston_run.

dataproc.req contains everything needed for classif_output_structs and Notebooks. 

Conflicting dependencies made it such that these had to be separate. 


# USEAGE

The main training program is neuston_net.py which has a convinient CLI interface.

```sh
$> ./neuston_net.py -h
usage: neuston_net.py [-h]
                      [--model {inception_v3,alexnet,squeezenet,vgg*,vgg*_bn,resnet*,,densenet*}]
                      [--pretrained] [--no-normalize]
                      [--max-epochs MAX_EPOCHS] [--min-epochs MIN_EPOCHS]
                      [--batch-size BATCH_SIZE] [--loaders LOADERS]
                      [--augment {flipx,flipy,flipxy,training-only,nochance} [...]]
                      [--learning-rate LEARNING_RATE]
                      training_dir evaluation_dir output_dir

positional arguments:
  training_dir          path to training set images
  evaluation_dir        path to testing set images
  output_dir            directory out output logs and saved model

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         select a model architecture to train. Options are:
                        {inception_v3,alexnet,squeezenet,vgg11,vgg11_bn,
                        vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,
                        resnet18,resnet34,resnet50,resnet101,resnet151,
                        densenet121,densenet169,densenet161,densenet201}
                        Default is inception_v3.
  --pretrained          Preloads model with weights trained on imagenet
  --no-normalize        If included, classes will NOT will be weighted during
                        training. Classes with fewer instances will not train
                        as well.
  --max-epochs EPOCHS   Maximum Number of Epochs. Training may
                        end before <max-epochs> if evaluation loss doesn't
                        improve beyond best_epoch*1.33. Default is 100.
  --min-epochs EPOCHS   Minimum number of epochs to run. Default is 10
  --batch-size SIZE     how many images to process in a batch. Default is 108, a
                        number divisible by 1,2,3,4 to ensure even batch
                        distribution across up to 4 GPUs.
  --loaders LOADERS     total number of threads to use for loading data
                        to/from GPUs. 4 per GPU is good. Default is 4 total.
  --augment AUG [AUG ...]
                        Data augmentation can improve training. Listed
                        transformations -may- be applied to any given input
                        image when loaded. Options are: {flipx,flipy,flipxy, 
                        training-only,nochance}. To deterministically apply
                        transformations to all input images while also
                        retaining non-transformed images, use "nochance".
                        flipx and flipy denotes mirroring the image vertically
                        and horizontally respectively. If "training-only" is
                        included, augmentation transformations will only be
                        applied to the training set.
  --learning-rate RATE, --lr RATE
                        The (initial) learning rate of the training optimizer
                        (Adam). Default is 0.001
```

