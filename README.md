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
`<DIR>` refers to the directory you choose to install into. “ifcb” is fine

1. `ssh <user>@poseidon.whoi.edu`
0. `cd $SCRATCH`
0. `git clone https://github.com/WHOIGit/ifcb_classifier.git <DIR>`
0. `cd <DIR>`
0. `conda env create -f environment.yaml`
    * If you get “conda: command not found”, the anaconda hpc Module may not be loaded. Do `module load anaconda` and try again.
     * Make sure you also have cuda modules loaded:
`module load cuda10.1/toolkit cuda10.1/cudnn/8 cuda10.1/blas cuda10.1/fft`
    * You can ensure modules load when you login with `module initadd <themodule>`
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
Created Model and validation results will be saved under `OUTDIR` as `model.ptl`
```sh
conda activate ifcbnn

## PARAMS ##
MODEL=inception_v3
DATASET=training-data/ExampleTrainingData
OUTDIR=training-output/ExampleTrainingResults
mkdir -vp "$OUTDIR"

python ./neuston_net.py TRAIN "$MODEL" "$DATASET" "$OUTDIR" 

```
Here are the default behaviors for the above command.

* `DATASET` is dynamically split into 80% training and 20% valdidating sub-datasets on a per-class basis. Dataset splitting is made repeatable thanks to the `--seed` flag. To fine tune `DATASET` see `--split`, `--class-config`, and `--class-min` 
* By default there is no data-augmentation. See `--flip`
* `MODEL` is pretrained and fully trainable. The output layer is automatically adjusted for the number of classes as determined by `DATASET` options.
* Epochs are limited to 10-60 epochs, with an early stopping criteria of 10 non-improving validation epochs. See `--emax` `--emin` `--estop`
* The command will output the following files to `OUTDIR`: 
  * model.ptl - The trained output model file. See `--model_file` flag
  * results.json - The results of the best validation epoch. See `--results` flag 
  * epochs.csv - a table of training_loss, validation_loss, and f1 scores for each epoch.
 
Below is the full set of options for the TRAIN command.
```sh
usage: neuston_net.py TRAIN [-h] [--untrain] [--img-norm MEAN STD] [--seed SEED] [--split T:V] [--class-config CSV COL] [--class-min MIN] [--emax MAX] [--emin MIN] [--estop STOP] [--flip {x,y,xy,x+V,y+V,xy+V}]
                            [--model-file MODEL_FILE] [--epochs-log EPOCHS_LOG] [--args-log ARGS_LOG] [--results FNAME [SERIES ...]]
                            MODEL SRC OUTDIR

positional arguments:
  MODEL                 Select a base model. Eg: "inception_v3"
  SRC                   Directory with class-label subfolders and images
  OUTDIR                Set output directory.

optional arguments:
  -h, --help            show this help message and exit

Model Adjustments:
  --untrain             If set, initializes MODEL ~without~ pretrained neurons. Default (unset) is pretrained
  --img-norm MEAN STD   Normalize images by MEAN and STD. This is like whitebalancing.

Dataset Adjustments:
  --seed SEED           Set a specific seed for deterministic output & dataset-splitting reproducability.
  --split T:V           Ratio of images per-class to split randomly into Training and Validation datasets. Randomness affected by SEED. Default is "80:20"
  --class-config CSV COL
                        Skip and combine classes as defined by column COL of a special CSV configuration file
  --class-min MIN       Exclude classes with fewer than MIN instances. Default is 2

Epoch Parameters:
  --emax MAX            Maximum number of training epochs. Default is 60
  --emin MIN            Minimum number of training epochs. Default is 10
  --estop STOP          Early Stopping: Number of epochs following a best-epoch after-which to stop training. Set STOP=0 to disable. Default is 10

Augmentation Options:
  Data Augmentation is a technique by which training results may improved by simulating novel input

  --flip {x,y,xy,x+V,y+V,xy+V}
                        Training images have 50% chance of being flipped along the designated axis: (x) vertically, (y) horizontally, (xy) either/both. May optionally specify "+V" to include Validation dataset

Output Options:
  --model-file MODEL_FILE
                        The file name of the output model. Default is model.ptl
  --epochs-log EPOCHS_LOG
                        Specify a csv filename. Includes epoch, loss, validation loss, and f1 scores. Default is epochs.csv
  --args-log ARGS_LOG   Specify a yaml filename. Includes all user-specified and default training parameters.
  --results FNAME [SERIES ...]
                        FNAME: Specify a validation-results filename or pattern. Valid patterns are: "{epoch}". Accepts .json .h5 and .mat file formats. Default is "results.json". SERIES: Options are: image_basenames,
                        image_fullpaths, output_ranks, output_fullranks, confusion_matrix. class_labels, input_classes, output_classes are always included by default. --results may be specified multiple times in order to
                        create different files. If not invoked, default is "results.json image_basenames"

```
## Model Running

```sh
conda activate ifcbnn

## PARAMS ##
MODEL=training-output/TrainedExample/model.ptl
DATASET=run-data/ExampleDataset
OUTDIR=run-output/ExampleRunResults
mkdir -vp "$OUTDIR"

python ./neuston_net.py RUN "$MODEL" "$DATASET" "$OUTDIR"

```
Here are the default behaviors for the above command.
* `DATASET` is assumed to be the path a directory containing bins. To instead operate on images directly, set `--type img`
* `MODEL` is the filepath a previously trained model file. 
* The command will output the following files to `OUTDIR`:
  * {bin}_class_v2.h5 - Classification result datafiles on a per bin basis, where `{bin}` is replaced by a given bin's ID. Several output formats are available, see `--outfile`

Additional flags for neuson_net.py RUN
```sh
## Defaults ##
usage: neuston_net.py RUN [-h] [--type {bin,img}] [--outfile OUTFILE]  [--filter IN|OUT KEYWORD [KEYWORD ...]] MODEL SRC OUTDIR

positional arguments:
  MODEL              Path to a previously-trained model file
  SRC                Resource(s) to be classified. Accepts a bin, an image, a text-file, or a directory. Directories are accessed recursively
  OUTDIR             Set output directory.

optional arguments:
  -h, --help         show this help message and exit
  --type {bin,img}   File type to perform classification on. Defaults is "bin"
  --outfile OUTFILE  Name/pattern of the output classification file. If TYPE==bin, "{bin}" in OUTFILE will be replaced with the bin id on a per-bin basis. If TYPE==img, "{dir}" in OUTFILE will be replaced with the parent
                     directory of classified images. A few output file formats are recognized: .csv, .mat, and .h5 (hdf). Default for TYPE==bin is "{bin}_class_v2.h5"; Default for TYPE==img is "{dir}.csv".
  --filter IN|OUT KEYWORD [KEYWORD ...]
                        Explicitly include (IN) or exclude (OUT) bins or image-files by KEYWORDs. KEYWORD may also be a text file containing KEYWORDs, line-deliminated.

```



