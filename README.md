# IFCB Classifier

This repo host an image classifying program designed to be trained on plankton images from an IFCB datasource.


# OUTLINE

**neuston_net.py** - Image classification neural net model trainer

**neuston_run.py** - Processes images using model produced by neuson_net

**plotutil.py** - module to assist with in-training plotting functions

**classif_output_structs.py** - module to help parse, process, plot the output of iterative neuston_net results 


# INSTALLATION (on WHOI HPC)

Installation and Setup on HPC
All installation commands are to be run from a terminal.
`<username>` refers to you whoi username and password
`<ifcb-dirname>` refers to the directory you choose to install into. “ifcb” is fine
`<ifcb-envname>` refers to the environment name you choose to install into. “ifcb” is fine

0. `ssh <username>@poseidon.whoi.edu`
0. `cd $SCRATCH`
0. `git clone https://github.com/WHOIGit/ifcb_classifier.git <ifcb-dirname>`
0. `git clone https://github.com/joefutrelle/pyifcb.git <ifcb-dirname>/pyifcb`
0. `cd <ifcb-dirname>`
0. `mkdir training-data training-output run-data run-output slurm-logs`
0. `conda create -n <ifcb-envname> python=3.7`
    * If you get “conda: command not found”, the anaconda hpc Module may not be loaded. Do `module load anaconda` and try again.
     * Make sure you also have cuda modules loaded:
`module load cuda91/toolkit cuda91/blas cuda91/cudnn cuda91/fft`
    * You can ensure modules load when you login with `module initadd <themodule>`
0. `conda activate <ifcb-envname>`
    * If you get prompted with “`conda init <SHELL_NAME>`” select “`bash`” for your shell-name. You will have to log out of poseidon (`exit`), log back in, and navigate back to this install directory.
0. `conda install pytorch=1.1 torchvision cudatoolkit=9.0 -c pytorch`
    * there are more recent versions of pytorch and cuda, but these are the ones that the hpc has cuda-version modules for.
0. `conda install scikit-learn matplotlib h5py scipy pandas requests scikit-image Pillow=6.1`
0. `pip install rectpack pysmb`
0. `(cd pyifcb; python setup.py install)`
0. DONE! Your installation is ready to go. You can test that things were installed correctly by doing `python neuston_net.py --help` and `python neuston_run.py --help`. The help screen/documentation for these scripts should appear.
0. You should optionally run the following command to bulk edit the example sbatch scripts to use your username. Replace YOURUSERNAME with your username.
    * `sed -i 's/username/YOURUSERNAME/g' batches/templates/example*`
 


# USAGE

Neuston Net and Neuston Run are the two main programs. They are for training and running models for plankton classification, respectively. 
Although the files under `batches/templates` are indended for use with the WHOI HPC's Slurm sbatch program, they describe typical usage well too.

## Neuston Net (Model Training)
Created Model will be saved under `OUTDIR` as `model.pt`
```sh
conda activate ifcb

## PARAMS ##
DATASET=training-data/ExampleTrainingData
OUTDIR=training-output/ExampleTrainingResults
mkdir -vp "$OUTDIR"

python ./neuston_net.py "$DATASET" "$OUTDIR" --split 80:20 --model inception_v3 

```

Additional useful flag for neuson_net:
```sh
## Flags to perhaps improve learning ##
# --pretrained   - start of the model architecture with weights based off of the model as trained on imagenet
# --augment      - augment the data to improve generalization
#                - eg: --augment flipxy training-only
# --wn           - perclass loss weight normalization 

## further configure what classes from DATASET to include/exclude/combine ##
# --class-config path/to/ExampleDataset-classlist-config.csv <column-header>
# --class-minimum N

## Additional flags and their default values ##
# --min-epochs 16
# --max-epochs 60
# --batch-size 108
# --loaders 4

## View the full list ##
# --help

```
## Neuston Run (Model Running)

```sh
conda activate ifcb

## PARAMS ##
MODEL=training-output/TrainedExample/model.pt
DATASET=run-data/ExampleDataset
OUTDIR=run-output/ExampleRunResults
mkdir -vp "$OUTDIR"

python ./neuston_run.py "$DATASET" --model "$MODEL" --outdir "$OUTDIR"

```
Additional flags for neuson_run:
```sh
## Defaults ##
# --input-type bins
# --outfile {bin}_class_v2.h5
# --batch-size 108
# --loaders 4

## Configure bin input ##
# --bin-filter path/to/your-list-of-bins.txt
# where "your-list-of-bins.txt" has one bin-id per line. eg: "IFCB5_2017_338_173613"
```



