# IFCB Classifier

This repo host an image classifying program designed to be trained on plankton images from an IFCB datasource.
This readme is written from the point of view of a WHOI user intent on installing this repo on the WHOI HPC.

# OUTLINE

**neuston_net.py** - Trains and runs ifcb classifiers

**sbatch/templates** - Contains slurm sbatch templates for training and running models on a slurm enabled HPC 

# INSTALLATION (on WHOI HPC)

Installation and Setup on HPC
All installation commands are to be run from a terminal.
`<user>` refers to your username (and password)

1. `ssh <user>@poseidon.whoi.edu`
0. `cd $SCRATCH`
0. `git clone https://github.com/WHOIGit/ifcb_classifier.git ifcbnn`
0. `cd ifcbnn`
0. `conda env create -f environment.yml`
    * If you get “conda: command not found”, the anaconda hpc Module may not be loaded. Do `module load anaconda` and try again.
    * You can ensure modules load when you login with `module initadd anaconda`
0. `conda activate ifcbnn`
    * If you get prompted with “`conda init <SHELL_NAME>`” select “`bash`” for your shell-name. You will have to log out of poseidon (`exit`), log back in, and navigate back to this DIR install directory.
0. DONE! Your installation is ready to go. You can test that things were installed correctly by doing `python neuston_net.py --help`. The help screen/documentation should appear.
0. You should optionally run the following command to bulk edit the example sbatch scripts to use your username. Replace YOURUSERNAME below with your username.
    * `sed -i 's/username/YOURUSERNAME/g' batches/templates/example*`


# USAGE

Neuston Net can be used for training and running image classification models.
Although the files under `batches/templates` are intended for use with the WHOI HPC's Slurm sbatch program, they describe typical usage well too.

DATASETs are directories containing class-label subfolders. Images must be located directly under each class-label subfolder. 
Training and Validation datasets are dynamically created from this one main DATASET folder.  

## Model Training
Created Model and validation results will be saved under `training-output/TRAIN_ID`. The model filetype is `.ptl`
```sh
conda activate ifcbnn

## PARAMS ##
TRAIN_ID=ExampleTrainingID
MODEL=inception_v3
DATASET=training-data/ExampleTrainingData

python neuston_net.py TRAIN "$TRAIN_ID" "$MODEL" "$DATASET"

```
Here are the default behaviors for the above command.

* `DATASET` is dynamically split into 80% training and 20% valdidating sub-datasets on a per-class basis. Dataset splitting is made repeatable thanks to the `--seed` flag. To fine tune `DATASET` see `--split`, `--class-config`, and `--class-min` 
* By default there is no data-augmentation. See `--flip`
* `MODEL` is pretrained and fully trainable. The output layer is automatically adjusted for the number of classes as determined by `DATASET` options.
* Epochs are limited to 10-60 epochs, with an early stopping criteria of 10 non-improving validation epochs. See `--emax` `--emin` `--estop`
* The command will output the following files to `training-output/TRAINING_ID` (see `--outdir`): 
  * TRAINING_ID.ptl - The trained output model file. See `--model-id` flag
  * results.json - The results of the best validation epoch. See `--results` flag 
  * epochs.csv - a table of training_loss, validation_loss, and f1 scores for each epoch.
  * hparams.yml - a human-readable file with the training input parameters and metadata (these are automatically baked into the model file too)
 
Below is the full set of options for the TRAIN command.
```sh
usage: neuston_net.py TRAIN [-h] [--untrain] [--img-norm MEAN STD] [--seed SEED] [--split T:V] [--class-config CSV COL]
                            [--class-min MIN] [--emax MAX] [--emin MIN] [--estop STOP] [--flip {x,y,xy,x+V,y+V,xy+V}]
                            [--outdir OUTDIR] [--model-id MODEL_ID] [--epochs-log EPOCHS_LOG] [--args-log ARGS_LOG]
                            [--results FNAME [SERIES ...]]
                            TRAINING_ID MODEL SRC

positional arguments:
  TRAINING_ID           Training ID. This value is the default value used by --outdir and --model-id.
  MODEL                 Select a base model. Eg: "inception_v3"
  SRC                   Directory with class-label subfolders and images. May also be a dataset-configuration csv.

optional arguments:
  -h, --help            show this help message and exit

Model Adjustments:
  --untrain             If set, initializes MODEL ~without~ pretrained neurons. Default (unset) is pretrained
  --img-norm MEAN STD   Normalize images by MEAN and STD. This is like whitebalancing.
                        eg1: "0.667 0.161", eg2: "0.056,0.058,0.051 0.067,0.071,0.057"

Dataset Adjustments:
  --seed SEED           Set a specific seed for deterministic output & dataset-splitting reproducability.
  --split T:V           Ratio of images per-class to split randomly into Training and Validation datasets. 
                        Randomness affected by SEED. Default is "80:20"
  --class-config CSV COL
                        Skip and combine classes as defined by column COL of a special CSV configuration file
  --class-min MIN       Exclude classes with fewer than MIN instances. Default is 2
  --class-max MAX       Limit classes to a MAX number of instances. 
                        If multiple datasets are specified with a dataset-configuration csv, 
                        classes from lower-priority datasets are truncated first. 

Epoch Parameters:
  --emax MAX            Maximum number of training epochs. Default is 60
  --emin MIN            Minimum number of training epochs. Default is 10
  --estop STOP          Early Stopping: Number of epochs following a best-epoch after-which to stop training. 
                        Set STOP=0 to disable. Default is 10

Augmentation Options:
  Data Augmentation is a technique by which training results may improved by simulating novel input

  --flip {x,y,xy,x+V,y+V,xy+V}
                        Training images have 50% chance of being flipped along the designated axis: 
                        (x) vertically, (y) horizontally, (xy) either/both. 
                        May optionally specify "+V" to include Validation dataset

Output Options:
  --outdir OUTDIR       Default is "training-output/{TRAINING_ID}"
  --model-id ID         Default is "{date}__{TRAINING_ID}"
  --epochs-log ELOG     Specify a csv filename. Includes epoch, loss, validation loss, and f1 scores. Default is epochs.csv
  --args-log ALOG       Specify a human-readable yaml filename. Includes all user-specified and default training parameters. Default is args.yml
  --results FNAME [SERIES ...]
                        FNAME: Specify a validation-results filename or pattern. Valid patterns are: "{epoch}". 
                               Accepts .json .h5 and .mat file formats.
                        SERIES: Data to include in FNAME. The following are always included and need not be specified: 
                                model_id, timestamp, class_labels, input_classes, output_classes.
                                Options are: image_basenames image_fullpaths
                                             output_scores output_winscores 
                                             confusion_matrix (ordered by classes_by_recall),
                                             classes_by_{count|f1|recall|precision}
                                             {f1|recall|precision}_{macro|weighted|perclass} 
                                             {counts|val_counts|train_counts}_perclass
                        --results may be specified multiple times in order to create different files. 
                        If not invoked, default is: "results.mat image_basenames output_scores counts_perclass confusion_matrix f1_perclass f1_weighted f1_macro"
```
## Model Running

```sh
conda activate ifcbnn

## PARAMS ##
RUN_ID=ExampleRunID
MODEL=training-output/TrainedExample/TrainedExample.ptl
DATASET=run-data/ExampleDataset

python neuston_net.py RUN "$RUN_ID" "$MODEL" "$DATASET"

```
Here are the default behaviors for the above command.
* `DATASET` is assumed to be the path a directory containing bins. To instead operate on images directly, set `--type img`
* `MODEL` is the filepath a previously trained model file. 
* The command will output the following files to `run-output/RUN_ID` (see `--outdir`):
  * {bin}_class_v2.h5 - Classification result datafiles on a per bin basis, where `{bin}` is replaced by a given bin's ID. Several output formats are available, see `--outfile`

Additional flags for neuson_net.py RUN
```sh
usage: neuston_net.py RUN [-h] [--type {bin,img}] [--outdir OUTDIR] [--outfile OUTFILE] [--filter IN|OUT [KEYWORD ...]]
                          RUN_ID MODEL SRC

positional arguments:
  RUN_ID                Run ID. Used by --outdir
  MODEL                 Path to a previously-trained model file
  SRC                   Resource(s) to be classified. Accepts a bin, an image, a text-file, or a directory. Directories are
                        accessed recursively

optional arguments:
  -h, --help            show this help message and exit
  --type {bin,img}      File type to perform classification on. Defaults is "bin"
  --outdir OUTDIR       Default is "run-output/{RUN_ID}"
  --outfile OUTFILE     Name/pattern of the output classification file. 
                        If TYPE==bin, "{bin}" in OUTFILE will be replaced with the bin id on a per-bin basis. 
                        A few output file formats are recognized: .json .mat .h5 (hdf).
                        Default for TYPE==bin is "{bin}_class_v2.h5"; Default for TYPE==img is "img_results.csv".
  --filter IN|OUT [KEYWORD ...]
                        Explicitly include (IN) or exclude (OUT) bins or image-files by KEYWORDs. 
                        KEYWORD may also be a text file containing KEYWORDs, line-deliminated.
  --clobber             If set, already processed bins in OUTDIR are reprocessed. 
                        By default, if an OUTFILE exists already the associated bin is not reprocessed.

```



