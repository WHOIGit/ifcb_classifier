"""this module handles the logging of data from epochs"""

import json
import os
import h5py as h5
import numpy as np
import pytorch_lightning as ptl
from scipy.io import savemat
import ifcb


## Training ##

def catch(func, handle=lambda e : e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)

class SaveValidationResults(ptl.callbacks.base.Callback):

    def __init__(self, outdir, outfile, series, best_only=True):
        self.outdir = outdir
        self.outfile = outfile
        self.best_only = best_only
        self.series = series

    def on_validation_end(self, trainer, pl_module):
        curr_epoch = pl_module.current_epoch
        class_labels = pl_module.args.classes
        input_classes = pl_module.validation_inputs
        image_fullpaths = pl_module.validation_input_srcs
        output_fullranks = pl_module.validation_outputs

        output_ranks = np.max(output_fullranks, axis=1)
        output_classes = np.argmax(output_fullranks, axis=1)
        image_basenames = [os.path.splitext(os.path.basename(img))[0] for img in image_fullpaths]
        confusion_matrix = None #TODO confusion_matrix, also must include recall-ordered class list (attrib for h5)

        assert output_fullranks.shape[0] == len(input_classes), 'wrong number inputs-to-outputs'
        assert output_fullranks.shape[1] == len(class_labels), 'wrong number of class labels'

        results = dict(class_labels=class_labels, input_classes=input_classes, output_classes=output_classes)

        if 'image_fullpaths' in self.series: results['image_fullpaths'] = image_fullpaths
        if 'image_basenames' in self.series: results['image_fullpaths'] = image_basenames
        if 'output_ranks' in self.series: results['output_ranks'] = output_ranks
        if 'output_fullranks' in self.series: results['output_fullranks'] = output_fullranks
        if 'confusion_matrix' in self.series : results['confusion_matrix'] = confusion_matrix

        outfile = os.path.join(self.outdir,self.outfile).format(epoch=curr_epoch)
        if curr_epoch==pl_module.best_epoch or not self.best_only:
            self.save_validation_results(outfile, results)

    def save_validation_results(self, outfile, results):
        if outfile.endswith('.json'): self.save_validation_results_json(outfile,results)
        if outfile.endswith('.mat'): self.save_validation_results_mat(outfile,results)
        if outfile.endswith('.h5'): self.save_validation_results_hdf(outfile,results)

    def save_validation_results_json(self,outfile,results):
        for np_series in ['input_classes','output_classes','output_ranks','output_fullranks','confusion_matrix']:
            if np_series in results: results[np_series] = results[np_series].tolist()
        with open(outfile, 'w') as f:
            json.dump(results, f)

    def save_validation_results_mat(self,outfile,results):
        for idx_series in ['input_classes','output_classes']:
            # matlab is not zero-indexed, so lets increment all the indicies by 1 >.>
            if idx_series in results: results[idx_series] += 1

        for float_series in ['output_ranks', 'output_fullranks', 'confusion_matrix']:
            if float_series in results: results[float_series] = results[float_series].astype('f4')
        for str_series in ['class_labels','image_fullpaths','image_basenames']:
            if str_series in results: results[str_series] = np.asarray(results[str_series], dtype='object')
        savemat(outfile, results, do_compression=True)

    def save_validation_results_hdf(self,outfile,results):
        with h5.File(outfile, 'w') as f:
            #meta = f.create_dataset('metadata', data=h5.Empty('f'))
            for num_series in ['input_classes','output_classes','output_ranks', 'output_fullranks', 'confusion_matrix']:
                if num_series in results:f.create_dataset(num_series, data=results[num_series], compression='gzip', dtype='float16')
            for str_series in ['class_labels','image_fullpaths','image_basenames']:
                if str_series in results: f.create_dataset(str_series, data=np.string_(results[str_series]), compression='gzip', dtype=h5.string_dtype())


## Running ##

class SaveRunResults(ptl.callbacks.base.Callback):
    pass
    '''
    def __init__(self, outdir, outfile, series):
        self.outdir = outdir
        self.outfile = outfile
        self.series = series

    def on_test_end(self, trainer, pl_module):
        classes = pl_module.args.classes
        input_images = pl_module.run_inputs
        output_ranks = pl_module.run_outputs
        output_classes = np.argmax(output_ranks, axis=1)

        assert output_ranks.shape[0] == len(input_images), 'wrong number inputs-to-outputs'
        assert outputs.shape[1] == len(classes), 'wrong number of class labels'

        results = dict(classes=classes, input_classes=input_classes, input_images=input_images,
                       output_classes=output_classes, output_ranks=output_ranks)

        outfile = os.path.join(self.outdir, self.outfile)
        if curr_epoch == pl_module.best_epoch or not self.best_only:
            self.save_run_results(outfile, results)

    def save_run_results(self, outfile, results):
        if outfile.endswith('.json'): self.save_run_results_json(outfile, results)
        if outfile.endswith('.mat'): self.save_run_results_mat(outfile, results)
        if outfile.endswith('.csv'): self.save_run_results_csv(outfile, results)
        if outfile.endswith('.h5'): self.save_run_results_hdf(outfile, results)
'''

def save_class_scores(path, bin_id, scores, roi_ids, classes):
    if path.endswith('.csv'):
        save_class_scores_csv(path, bin_id, scores, roi_ids, classes)
    elif path.endswith('.mat'):
        save_class_scores_mat(path, bin_id, scores, roi_ids, classes)
    else:  # hdf
        save_class_scores_hdf(path, bin_id, scores, roi_ids, classes)


def save_class_scores_hdf(path, bin_id, scores, roi_ids, classes):
    np_scores = np.array(scores)
    #print('length compare:  len(scores)==len(inputs)  -> {}=={} -> {}'.format(len(scores),len(roi_ids),len(scores)==len(roi_ids)))
    #print('class compare: len(score[0])==len(classes) -> {}=={} -> {}'.format(len(scores[0]),len(classes),len(scores[0])==len(classes)))
    assert np_scores.shape[0] == len(roi_ids), 'wrong number of ROI numbers'
    assert np_scores.shape[1] == len(classes), 'wrong number of class labels'
    roi_numbers = [ifcb.Pid(roi_id).target for roi_id in roi_ids]
    try:
        with h5.File(path, 'w') as f:
            ds = f.create_dataset('scores', data=np_scores, compression='gzip', dtype='float16')
            ds.attrs['bin_id'] = str(bin_id)
            ds.attrs['class_labels'] = [l.encode('ascii') for l in classes]
            f.create_dataset('roi_numbers', data=roi_numbers, compression='gzip', dtype='int16')
    except RuntimeError as e:
        # RuntimeError: Unable to create attribute (object header message is too large)
        # see: https://github.com/h5py/h5py/issues/1053#issuecomment-525363860
        print('   ', type(e), e, 'SWITCHING TO CSV OUTPUT')
        path = path.replace('.h5', '.csv').replace('.hdf', '.csv')
        save_class_scores_csv(path, bin_id, scores, roi_ids, classes)


def save_class_scores_csv(path, bin_id, scores, roi_ids, classes):
    #TODO: print lines if path == 'stdout'
    #roi_nums = [ifcb.Pid(roi_id).target for roi_id in roi_ids]
    roi_nums = roi_ids
    #print('length compare:  len(scores)==len(inputs)  -> {}=={} -> {}'.format(len(scores),len(roi_ids),len(scores)==len(roi_ids)))
    #print('class compare: len(score[0])==len(classes) -> {}=={} -> {}'.format(len(scores[0]),len(classes),len(scores[0])==len(classes)))

    with open(path, 'w') as f:
        header = str(bin_id)+',Highest_Ranking_Class,'+','.join(classes)
        f.write(header+os.linesep)
        for image_id, class_ranks_by_id in zip(roi_nums, scores):
            top_class = np.argmax(class_ranks_by_id)
            line = '{},{}'.format(image_id, classes[top_class])
            line_ranks = ','.join([str(r) for r in class_ranks_by_id])
            line = '{},{}'.format(line, line_ranks)
            f.write(line+os.linesep)


def save_class_scores_mat(path, bin_id, scores, roi_ids, classes):
    np_scores = np.array(scores)
    roi_numbers = [ifcb.Pid(roi_id).target for roi_id in roi_ids]
    d = {
        'version':     2,
        'roinum':      roi_numbers,
        'TBscores':    np_scores.astype('f4'),
        'class2useTB': np.asarray(classes, dtype='object')
    }
    savemat(path, d, do_compression=True)
