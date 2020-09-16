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
        #log: val_loss input_classes output_classes input_srcs outputs best epoch train_loss f1_macro f1_weighted loss f1_w f1_m best_ep'

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
        confusion_matrix = metrics.confusion_matrix(input_classes,output_classes,labels=classes_by['recall'], normalize=None)


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
        if 'output_winscores' in self.series: results['output_winscores'] = output_winscores
        if 'output_scores' in self.series: results['output_scores'] = output_scores
        if 'confusion_matrix' in self.series :
            results['confusion_matrix'] = confusion_matrix
            results['classes_by_recall'] = classes_by['recall'] # always include with confusion matrix
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
        for series in results: # numpy vals
            if isinstance(results[series], np.ndarray): results[series] = results[series].tolist()
        with open(outfile, 'w') as f:
            json.dump(results, f)

    def _save_validation_results_mat(self,outfile,results):
        # index ints
        idx_data =['input_classes','output_classes']
        idx_data.extend( ['classes_by_'+stat for stat in 'f1 recall precision count'.split()] )
        str_data = ['class_labels','image_fullpaths','image_basenames']

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
        int_data = ['input_classes', 'output_classes'] + 'counts_perclass val_counts_perclass train_counts_perclass'.split()
        int_data.extend(['classes_by_'+stat for stat in 'f1 recall precision count'.split()])
        string_data = ['class_labels', 'image_fullpaths', 'image_basenames']
        with h5.File(outfile, 'w') as f:
            meta = f.create_dataset('metadata', data=h5.Empty('f'))
            for series in results:
                if series in attrib_data: meta.attrs[series] = results[series]
                elif series in string_data: f.create_dataset(series, data=np.string_(results[series]), compression='gzip', dtype=h5.string_dtype())
                elif series in int_data: f.create_dataset(series, data=results[series], compression='gzip', dtype='int16')
                elif isinstance(results[series],np.ndarray):
                    f.create_dataset(series, data=results[series], compression='gzip', dtype='float16')
                else: print('hdf results: WE MISSED THIS ONE:',series)


## Running ##

class SaveRunResults(ptl.callbacks.base.Callback):

    def __init__(self, outdir, outfile, timestamp):
        self.outdir = outdir
        self.outfile = outfile
        self.timestamp = timestamp

    def on_test_end(self, trainer, pl_module):
        class_labels = pl_module.hparams.classes

        # TODO make sure this works as expected??
        rr = trainer.callback_metrics['RunResults']
        # RunResult: inputs, outputs, bin_id

        output_scores = rr.outputs

        output_classranks = np.max(output_scores, axis=1)
        output_classes = np.argmax(output_scores, axis=1)

        assert output_scores.shape[0] == len(output_classes), 'wrong number inputs-to-outputs'
        assert output_scores.shape[1] == len(class_labels), 'wrong number of class labels'

        results = dict(model_id=pl_module.hparams.model_id,
                       timestamp=self.timestamp,
                       class_labels=class_labels,
                       input_images=rr.inputs,
                       output_classes=output_classes,
                       output_scores=output_scores)

        outfile = os.path.join(self.outdir, self.outfile)
        dataset = pl_module.test_dataloader().dataset
        if isinstance(dataset, IfcbBinDataset):
            results['bin'] = str(dataset.bin.pid)
            results['roi_numbers'] = [ifcb.Pid(img).target for img in rr.inputs]
            outfile = outfile.format(bin=dataset.bin.pid)
        else: # ImageDataset
            pass
        os.makedirs(os.path.dirname(outfile),exist_ok=True)
        self.save_run_results(outfile, results)

    def save_run_results(self, outfile, results):
        if outfile.endswith('.json'): self._save_run_results_json(outfile, results)
        if outfile.endswith('.mat'): self._save_run_results_mat(outfile, results)
        if outfile.endswith('.h5'): self._save_run_results_hdf(outfile, results)

    def _save_run_results_json(self, outfile, results):
        # results: model_id timestamp class_labels (bin + roi_numbers)
        #          input_images output_classes output_scores

        output = dict(model_id = results['model_id'],
                      timestamp = results['timestamp'],
                      class_labels = results['class_labels'],
                      output_scores = results['output_scores'].tolist(),
                      output_classes = results['output_classes'].tolist() )
        if 'bin' in results:
            output['bin'] = results['bin']
            output['roi_numbers'] = results['roi_numbers']
        else:
            output['input_images'] = results['input_images']

        with open(outfile, 'w') as f:
            json.dump(output, f)

    def _save_run_results_mat(self, outfile, results):
        # results: model_id timestamp class_labels (bin + roi_numbers)
        #          input_images output_classes output_scores

        output=dict()
        # increment indexes, since matlab is not zero-indexed
        output['output_classes'] = results['output_classes'].astype('u4')+1
        output['model_id'] = results['model_id']
        output['timestamp'] = results['timestamp']
        output['output_scores'] = results['output_scores'].astype('f4')
        output['class_labels'] = np.asarray(results['class_labels'], dtype='object')
        if 'bin' in results:
            output['bin'] = results['bin']
            output['roi_numbers'] = results['roi_numbers']#.astype('u4') #not numpy yet
        else:
            output['input_images'] = np.asarray(results['input_images'], dtype='object')

        savemat(outfile, output, do_compression=True)

    def _save_run_results_hdf(self, outfile, results):
        # results: model_id timestamp class_labels (bin + roi_numbers)
        #          input_images output_classes output_scores

        with h5.File(outfile, 'w') as f:
            meta = f.create_dataset('metadata', data=h5.Empty('f'))
            meta.attrs['model_id'] = results['model_id']
            meta.attrs['timestamp'] = results['timestamp']
            f.create_dataset('output_classes', data=results['output_classes'], compression='gzip', dtype='float16')
            f.create_dataset('output_scores', data=results['output_scores'], compression='gzip', dtype='float16')
            f.create_dataset('class_labels', data=np.string_(results['class_labels']), compression='gzip', dtype=h5.string_dtype())
            if results['bin']:
                meta.attrs['bin'] = results['bin']
                f.create_dataset('roi_numbers', data=results['roi_numbers'], compression='gzip', dtype='float16')
            else:
                f.create_dataset('input_images', data=np.string_(results['input_images']), compression='gzip', dtype=h5.string_dtype())

