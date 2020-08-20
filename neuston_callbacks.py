"""this module handles the logging of data from epochs"""

# built in imports
import json
import os

# 3rd party imports
import h5py as h5
import numpy as np
import pytorch_lightning as ptl
from scipy.io import savemat

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
        curr_epoch = pl_module.current_epoch
        class_labels = pl_module.hparams.classes

        # TODO make sure this works as expected??
        log = trainer.callback_metrics # flattened dict
        #log: dict(val_loss input_classes output_classes input_srcs outputs best)
        #val_loss input_classes output_classes input_srcs outputs epoch best train_loss f1_macro f1_weighted loss f1_w f1_m best_ep'

        output_scores = log['outputs']
        image_fullpaths = log['input_srcs']

        output_winscores = np.max(output_scores, axis=1)
        output_classes = np.argmax(output_scores, axis=1)
        image_basenames = [os.path.splitext(os.path.basename(img))[0] for img in image_fullpaths]
        confusion_matrix = None #TODO confusion_matrix, also must include recall-ordered class list (attrib for h5)

        assert output_scores.shape[0] == len(log['input_classes']), 'wrong number inputs-to-outputs'
        assert output_scores.shape[1] == len(class_labels), 'wrong number of class labels'

        results = dict(model_id=pl_module.hparams.model_id,
                       timestamp=pl_module.hparams.cmd_timestamp,
                       class_labels=class_labels,
                       input_classes=log['input_classes'],
                       output_classes=output_classes)

        if 'image_fullpaths' in self.series: results['image_fullpaths'] = image_fullpaths
        if 'image_basenames' in self.series: results['image_basenames'] = image_basenames
        if 'output_winscores' in self.series: results['output_winscores'] = output_winscores
        if 'output_scores' in self.series: results['output_scores'] = output_scores
        if 'confusion_matrix' in self.series : results['confusion_matrix'] = confusion_matrix

        outfile = os.path.join(self.outdir,self.outfile).format(epoch=curr_epoch)
        if log['best'] or not self.best_only:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.save_validation_results(outfile, results)

    def save_validation_results(self, outfile, results):
        if outfile.endswith('.json'): self._save_validation_results_json(outfile,results)
        if outfile.endswith('.mat'): self._save_validation_results_mat(outfile,results)
        if outfile.endswith('.h5'): self._save_validation_results_hdf(outfile,results)

    def _save_validation_results_json(self,outfile,results):
        for np_series in ['input_classes','output_classes','output_scores','output_winscores','confusion_matrix']:
            if np_series in results: results[np_series] = results[np_series].tolist()
        with open(outfile, 'w') as f:
            json.dump(results, f)

    def _save_validation_results_mat(self,outfile,results):
        for idx_series in ['input_classes','output_classes']:
            # matlab is not zero-indexed, so lets increment all the indicies by 1 >.>
            if idx_series in results:
                results[idx_series] = results[idx_series].astype('u4') + 1

        for float_series in ['output_scores', 'output_winscores', 'confusion_matrix']:
            if float_series in results: results[float_series] = results[float_series].astype('f4')
        for str_series in ['class_labels','image_fullpaths','image_basenames']:
            if str_series in results: results[str_series] = np.asarray(results[str_series], dtype='object')
        savemat(outfile, results, do_compression=True)

    def _save_validation_results_hdf(self,outfile,results):
        with h5.File(outfile, 'w') as f:
            meta = f.create_dataset('metadata', data=h5.Empty('f'))
            meta.attrs['model_id'] = results['model_id']
            meta.attrs['timestamp'] = results['timestamp']
            for num_series in ['input_classes','output_classes','output_scores', 'output_winscores', 'confusion_matrix']:
                if num_series in results:f.create_dataset(num_series, data=results[num_series], compression='gzip', dtype='float16')
            for str_series in ['class_labels','image_fullpaths','image_basenames']:
                if str_series in results: f.create_dataset(str_series, data=np.string_(results[str_series]), compression='gzip', dtype=h5.string_dtype())


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

