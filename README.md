# IFCB Classifier

This repo host an image classifying program designed to be trained on plankton images from an IFCB datasource.

# INSTALLATION

The .req requirement files were generated with conda. Some of the libraries had to be installed with pip. These are denoted as such in the requirement files.

ifcb.req contains everything needed to run neuston net

tensorboard.req can be installed separately in a different conda environment. From this environment tensorboard can be launched to graphically view the plotting logs generated in neuston_net.py by tensorboardX.

# USEAGE
The main training program is neuston_net.py which has a convinient CLI interface.
```sh
$> ./neuston_net.py -h

positional arguments:
  training_dir          path to training set images
  testing_dir           path to testing set images
  output_dir            directory out output logs and saved model
  
  optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       How many epochs to run. May automatically stop before EPOCHS if validation loss plateaus but with do at least EPOCHS/2 runs. (default 10)
  --batch-size BATCH_SIZE
                        how many images to process in a batch, (default 108, a
                        number divisible by 1,2,3,4 to ensure even batch
                        distribution across up to 4 GPUs)
  --resume [RESUME]     path to a previously saved model to pick up from
                        (defaults to <output_dir>)
  --model {inception_v3,alexnet,squeezenet,vgg*,vgg*_bn,resnet*,densenet*}
                        select a model (default to inceptin_v3)
```
