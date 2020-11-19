"""this module handles the logging of data from epochs"""

# built in imports
import json
import os

# 3rd party imports
import h5py as h5
import numpy as np
import pytorch_lightning as ptl
from scipy.io import savemat
from sklearn import metrics

# project imports
import ifcb
from neuston_data import IfcbBinDataset

## Training ##

class SaveValidationResults(ptl.callbacks.base.Callback):

    def __init__(self, outdir, outfile, series, best_only=True):
        self.outdir = outdir
        self.outfile = outfile
        self.series = series
        self.best_only = best_only

    def on_validation_end(self, trainer, pl_module):
        log = trainer.callback_metrics # flattened dict
        #log: val_loss input_classes output_classes input_srcs outputs epoch best train_loss f1_macro f1_weighted

        if not(log['best'] or not self.best_only):
            return

        curr_epoch = pl_module.current_epoch
        class_labels = pl_module.hparams.classes
        class_idxs = list(range(len(class_labels)))

        val_dataset = pl_module.val_dataloader().dataset
        train_dataset = pl_module.train_dataloader().dataset
        val_counts_perclass = val_dataset.count_perclass
        train_counts_perclass = train_dataset.count_perclass
        counts_perclass = [vcount+tcount for vcount,tcount in zip(val_counts_perclass, train_counts_perclass)] # element-wise addition
        training_image_fullpaths = train_dataset.images
        training_image_basenames = [os.path.splitext(os.path.basename(img))[0] for img in training_image_fullpaths]
        training_classes = train_dataset.targets

        output_scores = log['outputs']
        output_winscores = np.max(output_scores, axis=1)
        output_classes = np.argmax(output_scores, axis=1)
        input_classes = log['input_classes']
        image_fullpaths = log['input_srcs']
        image_basenames = [os.path.splitext(os.path.basename(img))[0] for img in image_fullpaths]

        assert output_scores.shape[0] == len(input_classes), 'wrong number inputs-to-outputs'
        assert output_scores.shape[1] == len(class_labels), 'wrong number of class labels'

        # STATS!
        stats = dict()
        for mode in ['weighted','macro', None]:
            for stat in ['f1','recall','precision']:
                metric = getattr(metrics,stat+'_score')(input_classes,output_classes,labels=class_idxs,average=mode, zero_division=0)
                label = '{}_{}'.format(stat,mode if mode else 'perclass')
                stats[label] = metric  # f1|recall|precision _ macro|weighted|perclass

        # Classes order by some Stat
        classes_by = dict()
        classes_by['count'] = sorted(class_idxs, key=lambda idx: (counts_perclass[idx]), reverse=True) #higher is better
        for stat in ['f1','recall','precision']:
            classes_by[stat] = sorted(class_idxs, key=lambda idx: (stats[stat+'_perclass'][idx]), reverse=True)

        # Confusion matrix
        #confusion_matrix = metrics.confusion_matrix(input_classes,output_classes,labels=classes_by['recall'], normalize=None)
        confusion_matrix = metrics.confusion_matrix(input_classes, output_classes, labels=class_idxs, normalize=None)

        ## PASSING IT DOWN TO OUTPUTS ##

        # default values
        results = dict(model_id=pl_module.hparams.model_id,
                       timestamp=pl_module.hparams.cmd_timestamp,
                       class_labels=class_labels,
                       input_classes=input_classes,
                       output_classes=output_classes)

        # optional values
        if 'image_fullpaths' in self.series: results['image_fullpaths'] = image_fullpaths
        if 'image_basenames' in self.series: results['image_basenames'] = image_basenames
        if 'training_image_fullpaths' in self.series: results['training_image_fullpaths'] = training_image_fullpaths
        if 'training_image_basenames' in self.series: results['training_image_basenames'] = training_image_basenames
        if 'training_classes' in self.series: results['training_classes'] = training_classes
        if 'output_winscores' in self.series: results['output_winscores'] = output_winscores
        if 'output_scores' in self.series: results['output_scores'] = output_scores
        if 'confusion_matrix' in self.series :
            results['confusion_matrix'] = confusion_matrix
            #results['classes_by_recall'] = classes_by['recall'] # no longer included with cm
        if 'counts_perclass' in self.series: results['counts_perclass'] = counts_perclass
        if 'val_counts_perclass' in self.series: results['val_counts_perclass'] = val_counts_perclass
        if 'train_counts_perclass' in self.series: results['val_counts_perclass'] = val_counts_perclass

        # optional stats and class_by's
        for stat in stats: # eg f1_weighted, recall_perclass
            if stat in self.series: results[stat] = stats[stat]
        for stat in classes_by:
            classes_by_stat = 'classes_by_'+stat
            if classes_by_stat in self.series: results[classes_by_stat] = classes_by[stat]

        # sendit!
        outfile = os.path.join(self.outdir,self.outfile).format(epoch=curr_epoch)
        if log['best'] or not self.best_only:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.save_validation_results(outfile, results)

    def save_validation_results(self, outfile, results):
        if outfile.endswith('.json'): self._save_validation_results_json(outfile,results)
        if outfile.endswith('.mat'): self._save_validation_results_mat(outfile,results)
        if outfile.endswith('.h5'): self._save_validation_results_hdf(outfile,results)

    def _save_validation_results_json(self,outfile,results):
        for series in results: # convert numpy arrays to list
            if isinstance(results[series], np.ndarray):
                results[series] = results[series].tolist()
        # write json file
        with open(outfile, 'w') as f:
            json.dump(results, f)

    def _save_validation_results_mat(self,outfile,results):
        # index ints
        idx_data = ['input_classes','output_classes','training_classes']
        idx_data += ['classes_by_'+stat for stat in 'f1 recall precision count'.split()]
        str_data = ['class_labels','image_fullpaths','image_basenames','training_image_fullpaths','training_image_basenames']

        for series in results:
            if isinstance(results[series], np.ndarray): results[series] = results[series].astype('f4')
            elif isinstance(results[series], np.float64): results[series] = results[series].astype('f4')
            elif series in str_data: results[series] = np.asarray(results[series], dtype='object')
            elif series in idx_data: results[series] = np.asarray(results[series]).astype('u4') + 1
            # matlab is not zero-indexed, so increment all the indicies by 1

        savemat(outfile, results, do_compression=True)

    def _save_validation_results_hdf(self,outfile,results):
        attrib_data = ['model_id', 'timestamp']
        attrib_data += 'f1_weighted recall_weighted precision_weighted f1_macro recall_macro precision_macro'.split()
        int_data = ['input_classes', 'output_classes', 'training_classes']
        int_data += 'counts_perclass val_counts_perclass train_counts_perclass'.split()
        int_data += ['classes_by_'+stat for stat in 'f1 recall precision count'.split()]
        string_data = ['class_labels', 'image_fullpaths', 'image_basenames', 'training_image_fullpaths', 'training_image_basenames']
        with h5.File(outfile, 'w') as f:
            meta = f.create_dataset('metadata', data=h5.Empty('f'))
            for series in results:
                if series in attrib_data: meta.attrs[series] = results[series]
                elif series in string_data: f.create_dataset(series, data=np.string_(results[series]), compression='gzip', dtype=h5.string_dtype())
                elif series in int_data: f.create_dataset(series, data=results[series], compression='gzip', dtype='int16')
                elif isinstance(results[series],np.ndarray):
                    f.create_dataset(series, data=results[series], compression='gzip', dtype='float16')
                else: raise UserWarning('hdf results: WE MISSED THIS ONE: {}'.format(series))


## Running ##
def save_run_results(input_images, output_scores, class_labels, timestamp, outdir, outfile, model_id=None, input_obj=None):
    output_classranks = np.max(output_scores, axis=1)
    output_classes = np.argmax(output_scores, axis=1)

    assert output_scores.shape[0] == len(output_classes), 'wrong number inputs-to-outputs'
    assert output_scores.shape[1] == len(class_labels), 'wrong number of class labels'

    results = dict(version='v3',
                   model_id=model_id,
                   timestamp=timestamp,
                   class_labels=class_labels,
                   input_images=input_images,
                   output_classes=output_classes,
                   output_scores=output_scores)

    outfile = os.path.join(outdir, outfile)
    if isinstance(input_obj,ifcb.Pid):
        bin_obj = input_obj
        results['bin_id'] = bin_obj.pid
        results['roi_numbers'] = [ifcb.Pid(img).target for img in input_images]
        outfile_dict = dict(BIN_ID=bin_obj.pid, INPUT_SUBDIRS=bin_obj.namespace,
                            BIN_YEAR=bin_obj.year, BIN_DATE=bin_obj.yearday)
        outfile = outfile.format(**outfile_dict).replace(2*os.sep,os.sep)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        _save_run_results(outfile, results)
    else:  # ImageDataset
        if '{INPUT_SUBDIRS}' in outfile:
            dir_groups = {}
            input_src = input_obj if os.path.isdir(input_obj) else ''
            for img_path,img_classidx,img_scores in zip(input_images,output_classes,output_scores):
                parent_dir = os.path.dirname(img_path.replace(input_src, ''))+os.sep

                if parent_dir not in dir_groups:
                    dir_groups[parent_dir] = {k: v if k not in ['input_images','output_classes','output_scores'] else [] for k,v in results.items()}
                dir_groups[parent_dir]['input_images'].append(os.path.basename(img_path))
                dir_groups[parent_dir]['output_classes'].append(img_classidx)
                dir_groups[parent_dir]['output_scores'].append(img_scores)
            for parent_dir,sub_results in dir_groups.items():
                sub_outfile = outfile.format(INPUT_SUBDIRS=parent_dir)
                os.makedirs(os.path.dirname(sub_outfile),exist_ok=True)
                sub_results['output_classes'] = np.asarray(sub_results['output_classes'], dtype=results['output_classes'].dtype)
                sub_results['output_scores'] = np.asarray(sub_results['output_scores'], dtype=results['output_scores'].dtype)
                _save_run_results(sub_outfile, sub_results)

        else: #easy
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            _save_run_results(outfile, results)


def _save_run_results(outfile, results):
    # handles .json, .mat, .h5 files
    ext = os.path.splitext(outfile)[-1]
    assert ext in ['.json','.mat','.h5'], 'output fileformat "{}" not valid'.format(ext)
    def _save_run_results_json(outfile, results):
        # results: model_id timestamp class_labels (bin + roi_numbers)
        #          input_images output_classes output_scores

        output = dict(version = results['version'],
                      model_id = results['model_id'],
                      timestamp = results['timestamp'],
                      class_labels = results['class_labels'],
                      output_scores = results['output_scores'].tolist(),
                      output_classes = results['output_classes'].tolist() )
        if 'bin_id' in results:
            output['bin_id'] = results['bin_id']
            output['roi_numbers'] = results['roi_numbers']
        else:
            output['input_images'] = results['input_images']

        with open(outfile, 'w') as f:
            json.dump(output, f)

    def _save_run_results_mat(outfile, results):
        # results: model_id timestamp class_labels (bin + roi_numbers)
        #          input_images output_classes output_scores

        output=dict()
        # increment indexes, since matlab is not zero-indexed
        output['output_classes'] = results['output_classes'].astype('u4')+1
        output['version'] = results['version']
        output['model_id'] = results['model_id']
        output['timestamp'] = results['timestamp']
        output['output_scores'] = results['output_scores'].astype('f4')
        output['class_labels'] = np.asarray(results['class_labels'], dtype='object')
        if 'bin_id' in results:
            output['bin_id'] = results['bin_id']
            output['roi_numbers'] = results['roi_numbers']#.astype('u4') #not numpy yet, but thats fine it seems
        else:
            output['input_images'] = np.asarray(results['input_images'], dtype='object')

        savemat(outfile, output, do_compression=True)

    def _save_run_results_hdf(outfile, results):
        # results: model_id timestamp class_labels (bin + roi_numbers)
        #          input_images output_classes output_scores

        with h5.File(outfile, 'w') as f:
            meta = f.create_dataset('metadata', data=h5.Empty('f'))
            meta.attrs['version'] = results['version']
            meta.attrs['model_id'] = results['model_id']
            meta.attrs['timestamp'] = results['timestamp']
            f.create_dataset('output_classes', data=results['output_classes'], compression='gzip', dtype='float16')
            f.create_dataset('output_scores', data=results['output_scores'], compression='gzip', dtype='float16')
            f.create_dataset('class_labels', data=np.string_(results['class_labels']), compression='gzip', dtype=h5.string_dtype())
            if results['bin_id']:
                meta.attrs['bin_id'] = results['bin_id']
                f.create_dataset('roi_numbers', data=results['roi_numbers'], compression='gzip', dtype='uint16')
            else:
                f.create_dataset('input_images', data=np.string_(results['input_images']), compression='gzip', dtype=h5.string_dtype())

    if outfile.endswith('.json'): _save_run_results_json(outfile, results)
    if outfile.endswith('.mat'): _save_run_results_mat(outfile, results)
    if outfile.endswith('.h5'): _save_run_results_hdf(outfile, results)


class SaveTestResults(ptl.callbacks.base.Callback):

    def __init__(self, outdir, outfile, timestamp):
        self.outdir = outdir
        self.outfile = outfile
        self.timestamp = timestamp

    def on_test_end(self, trainer, pl_module):

        RRs = trainer.callback_metrics['RunResults']
        # RunResult rr: inputs, outputs, bin_id
        if not isinstance(RRs,list):
            RRs = [RRs]

        for rr in RRs:
            input_obj = rr.input_obj
            output_scores = rr.outputs
            input_images = rr.inputs
            model_id = pl_module.hparams.model_id
            class_labels = pl_module.hparams.classes

            save_run_results(input_images, output_scores, class_labels, self.timestamp, self.outdir, self.outfile, model_id, input_obj)
