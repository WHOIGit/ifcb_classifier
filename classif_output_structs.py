import os
from ast import literal_eval
from sklearn import metrics
import pandas as pd
from statistics import stdev, mean

import plotutil


class Epoch:
    """A Single epoch of a classification-training run"""
    def __init__(self,epoch_dict):

        if isinstance(epoch_dict,str) and os.path.isfile(epoch_dict):
            with open(epoch_dict) as f:
                epoch_dict = literal_eval(f.read())

        ##catch-all
        #for k in epoch_dict:
        #    setattr(self, k, epoch_dict[k])
        ##but, to be explicit...

        # epoch results data
        self.true_inputs        = epoch_dict['true_inputs']
        self.true_images        = epoch_dict['true_images']
        self.prediction_outputs = epoch_dict['prediction_outputs']

        # Epoch metadata
        self.epoch_num      = epoch_dict['epoch']
        self.classes        = epoch_dict['classes']
        self.eval_loss      = epoch_dict['eval_loss']
        self.training_loss  = epoch_dict['training_loss']
        self.secs_elapsed   = epoch_dict['secs_elapsed']

        # Run/Input metadata
        self.name           = epoch_dict['name']
        self.output_dir     = epoch_dict['output_dir']

        # Series/Input metadata
        self.model          = epoch_dict['model']
        self.training_dir   = epoch_dict['training_dir']
        self.eval_dir       = epoch_dict['evaluation_dir']
        self.pretrained     = epoch_dict['pretrained']
        self.normalized     = not epoch_dict['no_normalize']
        self.min_epochs     = epoch_dict['min_epochs']
        self.max_epochs     = epoch_dict['max_epochs']
        self.batch_size     = epoch_dict['batch_size']
        self.loaders        = epoch_dict['loaders']
        self.augments       = epoch_dict['augment']
        self.learning_rate  = epoch_dict['learning_rate']

        # derived values, non proc intensive. otherwise see __getattr__
        self.hours_elapsed = round(self.secs_elapsed/60/60, 2)
        self.mins_per_epoch = round(self.secs_elapsed/60/self.epoch_num, 2)

    @property
    def input_labels(self): return [self.classes[i] for i in self.true_inputs]

    @property
    def output_labels(self): return [self.classes[i] for i in self.prediction_outputs]

    def __getattr__(self, name: str):
        """Returns the attribute matching passed name."""
        
        # Get internal dict value matching name.
        value = self.__dict__.get(name)
        
        # if value exists, return it now
        if value is not None: return value
        # otherwise:

        ## create stats on the fly! ##
        args = [self.true_inputs, self.prediction_outputs]
        if name == 'accuracy':    self.accuracy = metrics.accuracy_score(*args)

        # f1
        elif name == 'f1_weighted': self.f1_weighted = metrics.f1_score(*args,average='weighted')
        elif name == 'f1_macro':    self.f1_macro    = metrics.f1_score(*args,average='macro')
        elif name == 'f1_micro':    self.f1_micro    = metrics.f1_score(*args,average='micro')

        # recall
        elif name == 'recall_weighted': self.recall_weighted = metrics.recall_score(*args,average='weighted')
        elif name == 'recall_macro':    self.recall_macro    = metrics.recall_score(*args,average='macro')
        elif name == 'recall_micro':    self.recall_micro    = metrics.recall_score(*args,average='micro')

        # precision
        elif name == 'precision_weighted': self.precision_weighted = metrics.precision_score(*args,average='weighted')
        elif name == 'precision_macro':    self.precision_macro    = metrics.precision_score(*args,average='macro')
        elif name == 'precision_micro':    self.precision_micro    = metrics.precision_score(*args,average='micro')

        # perclass
        elif name == 'f1_perclass':        self.f1_perclass        = metrics.f1_score(*args,average=None)
        elif name == 'recall_perclass':    self.recall_perclass    = metrics.recall_score(*args,average=None)
        elif name == 'precision_perclass': self.precision_perclass = metrics.precision_score(*args,average=None)
        elif name == 'count_perclass':     self.count_perclass     = [self.true_inputs.count(i) for i,c in enumerate(self.classes)]

        else:
            # Raise AttributeError if attribute value not found and is not to be created on the fly.
            raise AttributeError(f'{self.__class__.__name__}.{name} is invalid.')
        brand_new_value = getattr(self,name)
        return brand_new_value


    @classmethod
    def load(cls,epoch_dict_fpath):
        with open(epoch_dict_fpath) as f:
             return cls(literal_eval(f.read()))


    def get_input_metadata(self):
        input_metadata = {}
        for a in Run.metadata_attribs:
            input_metadata[a] = self.__getattr__(a)
        return input_metadata
    def get_series_metadata(self):
        series_metadata = {}
        for a in Series.metadata_attribs:
            series_metadata[a] = self.__getattr__(a)
        return series_metadata


    def plot_confusion_matrix(self, title=None, output='show', sort_by='recall'):
        if title is None:
            title = '{} (epoch={}) f1_w={:.1f}%'.format(self.name,self.epoch_num,100*self.f1_weighted)

        if sort_by == 'alphabetical':
            order = sorted(self.classes)
        elif sort_by == 'recall':
            order = sorted(self.classes, reverse=True,
                     key=lambda c: (self.recall_perclass[self.classes.index(c)],
                                    self.f1_perclass[self.classes.index(c)]))
        plotutil.make_confusion_matrix_plot(self.input_labels, 
                                            self.output_labels, 
                                            order,
                                            title=title,
                                            normalize_mapping=True,
                                            text_as_percentage=True,
                                            output=output)

    def perclass_barplot(self, stat='f1', start=None, end=1, sort_by=['max','min','alphabetical','count'][1]):
        if stat == 'f1': stat_att = 'f1_perclass'
        elif stat == 'recall': stat_att = 'recall_perclass'
        elif stat == 'precision': stat_att = 'precision_perclass'

        #TODO sorting

        stat_dict = {'classes':self.classes, stat:getattr(self,stat_att), 'counts':self.count_perclass}
        df = pd.DataFrame(stat_dict)
        df = df.sort_values(by=stat)
        
        ax = df.plot.barh(x='classes', y=stat, figsize=[12, len(self.classes)/3])
        ax.set_title('Perclass {} for {}'.format(stat,self.name))
        ax.grid(True, which='both', axis='x')
        ax.set_xlim(start,end)
        return ax

    def pairwise_df(self, skip_correct=False, skip_empties=False ):

        cm = metrics.confusion_matrix(self.input_labels,self.output_labels,self.classes)

        df_matrix = pd.DataFrame(cm,index=self.classes,columns=self.classes)
        df_matrix.index.name = 'Input'
        df_matrix.columns.name = 'Output'
        df = df_matrix.stack().rename("count").reset_index()

        # ci=class input, cp=class predicted
        pairwise_metadata = {ci:{cp:{'images':[]} for cp in self.classes} for ci in self.classes}
        combo = zip(self.input_labels,self.true_images,self.output_labels)
        for input_label,input_image,output_label in combo:
            pairwise_metadata[input_label][output_label]['images'].append(input_image)

        def add_images(row):
            input_label = row[0]
            output_label = row[1]
            images = pairwise_metadata[input_label][output_label]['images']
            return images, len(images)

        df['images'],df['count'] = zip(*df.apply(add_images, axis=1))

        if skip_correct:
            df = df[df['Input'] != df['Output']]
        if skip_empties:
            df = df[df['count'] != 0]

        df = df.set_index(['Input', 'Output'])

        return df

class ComboEpoch(Epoch):

    def __init__(self, epochs):

        self.true_inputs=[]
        self.true_images=[]
        self.prediction_outputs=[]
        self.classes = None
        self.epochs = []
        for e in epochs:
            if isinstance(e, str) and e.endswith('.dict'):
                e = Epoch.load(e)                
            elif not isinstance(e,Epoch):
                raise ValueError('"epochs" must be a list of Epoch, or a list of string filepaths ending in .dict')
            self.epochs.append(e)
            self.true_inputs.extend(e.true_inputs)
            self.true_images.extend(e.true_images)
            self.prediction_outputs.extend(e.prediction_outputs)
            if self.classes:
                # on subsequent passes, makes sure classes lists are identical
                assert self.classes == e.classes
            # on first pass, set classes
            else: self.classes = e.classes

    def get_input_metadata(self):
        raise NotImplementedError

    def plot_confusion_matrix(self, title=None, output='show', order='recall'):
        if title is None:
            title = '{}x Combo Confusion Matrix F1_w={:.1f}%'.format(len(self.names),100*self.f1_weighted)
        super().plot_confusion_matrix(title,output,order)

    def pairwise_df(self, skip_correct=True, skip_empties=True, keep_dupes=None, naughty_sort=False):

        df = super().pairwise_df(skip_correct=skip_correct, skip_empties=skip_empties)

        if keep_dupes is None:
            keep_dupes = range(1,len(self.epochs)+1)
        elif isinstance(keep_dupes,int):
            keep_dupes = [keep_dupes]
        elif isinstance(keep_dupes,list): pass
        elif keep_dupes.startswith('>='):
            value = int(keep_dupes.replace('>=',''))
            keep_dupes = range(value,len(self.epochs)+1)
        elif keep_dupes.startswith('>'):
            value = int(keep_dupes.replace('>',''))
            keep_dupes = range(value+1,len(self.epochs)+1)
        elif '-' in keep_dupes:
            v1,v2 = keep_dupes.split('-')
            v1,v2 = int(v1),int(v2)
            keep_dupes = range(v1,v2+1)
        else:
            try: keep_dupes = [int(keep_dupes)]
            except: raise ValueError(keep_dupes,'not valid')
        keep_dupes = list(keep_dupes)

        def denote_duplicates(row):

            image_set = sorted(set(row['images']))
            image_set_counts = [len([i for i in row['images'] if i == img]) for img in image_set]

            # sort images by most repeated to least
            if image_set:  # doesn't work for empty lists
                count_image_tups = sorted(zip(image_set_counts, image_set), reverse=True)
                count_image_tups = [tup for tup in count_image_tups if tup[0] in keep_dupes]
                if len(count_image_tups) == 0: return [],[],0
                image_set_counts, image_set = list(zip(*count_image_tups))
                image_set_counts, image_set = list(image_set_counts), list(image_set)

            return image_set, image_set_counts, len(image_set)

        df['images'], df['image_counts'], df['count'] = zip(*df.apply(denote_duplicates, axis=1))
        if skip_empties:
            df = df[df['count'] != 0]

        if naughty_sort:
            df['naughty_rank'] = df.image_counts.apply(lambda counts: sum(c**2 for c in counts))
            df = df.sort_values('naughty_rank',ascending=False)
            df = df.drop('naughty_rank',1)

        return df

    def naughty_dupes(self,minimum=1,plot=False):
        dupes = range(minimum, len(self.epochs)+1)
        dupe_vals = [self.pairwise_df(keep_dupes=dupe)['count'].sum() for dupe in dupes]
        df = pd.DataFrame(dict(dupes=dupes,count=dupe_vals))

        if plot:
            ax = df.plot.bar(x='dupes',y='count',title='Histogram of Chronically Misclassified Images')
            return ax
        else:
            df.set_index('dupes', inplace=True)
            return df

    @property
    def names(self): return [e.name for e in self.epochs]

    @property
    def output_dirs(self): return [e.output_dirs for e in self.epochs]

    @property
    def epoch_nums(self):
        return [e.epoch_nums for e in self.epochs]

    def __len__(self):
        return len(self.epochs)
    
    
    


class Run:
    """A single classification-training run's results"""

    metadata_attribs = ['name', 'classes', 'output_dir'] + \
                       ['model',
                        'training_dir', 'eval_dir',
                        'pretrained', 'normalized',
                        'augments', 'learning_rate',
                        'min_epochs', 'max_epochs',
                        'batch_size', 'loaders']
    # + Series.metadata_attribs

    def __init__(self, evaluation_records:str):
        self.src = evaluation_records

        self.epochs = []
        with open(evaluation_records) as f:
            epoch_dicts = literal_eval(f.read())
            for epoch_dict in epoch_dicts:
                self.epochs.append( Epoch(epoch_dict) )

        self.best_epoch_num = 0
        self.best_eval_loss = float('inf')
        for epoch in self.epochs:
            if epoch.eval_loss < self.best_eval_loss:
                self.best_epoch_num = epoch.epoch_num
                self.best_eval_loss = epoch.eval_loss
        self.best_epoch = self.epochs[self.best_epoch_num]

        for att in Run.metadata_attribs:
            setattr(self, att, getattr(self.best_epoch,att))


    def get_series_metadata(self):
        series_metadata = self.best_epoch.get_series_metadata()
        #series_metadata.pop('name')
        #series_metadata.pop('output_dir')
        return series_metadata

    def __len__(self):
        return len(self.epochs)

    def duration_seconds(self):
        return self.epochs[-1].secs_elapsed

    def plot_loss(self, output='show', normalize=True):
        train_losses = [e.training_loss for e in self.epochs]
        eval_losses = [e.eval_loss for e in self.epochs]
        loss_tups = list(zip(train_losses,eval_losses))
        title = '{} Loss{}'.format(self.name, ' (Normalized)' if normalize else '')
        plotutil.loss(loss_tups,output,title,normalize)


class Series:
    """A set of runs with identical input parameters at runtime but independent results"""

    metadata_attribs = Run.metadata_attribs.copy()
    metadata_attribs.remove('name')
    metadata_attribs.remove('output_dir')

    def __init__(self, series_dir, best_epochs_only=False):
        self.src = series_dir
        self.name = os.path.basename(series_dir)

        runs_or_epochs = []
        subdirs = [os.path.join(series_dir, subdir) for subdir in os.listdir(series_dir)]
        for subdir in subdirs:
            if os.path.isdir(subdir):
                if best_epochs_only:
                    run_or_epoch = os.path.join(subdir, 'best_epoch.dict')
                    run_or_epoch = Epoch.load(run_or_epoch)
                else:
                    run_or_epoch = os.path.join(subdir,'evaluation_records.lod')
                    run_or_epoch = Run( run_or_epoch )
                if runs_or_epochs:
                    assert runs_or_epochs[-1].get_series_metadata() == run_or_epoch.get_series_metadata()
                runs_or_epochs.append(run_or_epoch)
        if best_epochs_only:
            self.best_epochs = runs_or_epochs
            self.runs = NotImplemented
        else:
            self.runs = runs_or_epochs
            self.best_epochs = [run.best_epoch for run in self.runs]

        # series metadata
        for att in Series.metadata_attribs:
            setattr(self, att, getattr(self.best_epochs[0],att))

    def __len__(self):
        return len(self.best_epochs)

    def boxplot(self, input_stats='all', start=0.75, end=1):

        if input_stats == 'weighted':
            stats = 'f1_weighted recall_weighted precision_weighted'.split()
        elif input_stats == 'macro':
            stats = 'f1_macro recall_macro precision_macro'.split()
        elif input_stats == 'all':
            stats = 'accuracy f1_weighted f1_macro recall_weighted recall_macro precision_weighted precision_macro'.split()
        else:
            stats = input_stats

        stat_dict = {stat: [getattr(be, stat) for be in self.best_epochs] for stat in stats}

        df = pd.DataFrame(stat_dict)
        ax = df.plot(kind='box', vert=False, figsize=[12, len(stats)/2])
        ax.set_title('{} Stats for {}'.format(input_stats.capitalize(), self.name))
        ax.grid(True, which='both', axis='x')
        ax.set_xlim(start, end)
        return ax

    def perclass_boxplot(self, stat=['f1', 'recall', 'precision'][0], sort_by=[mean, min, max, stdev, 'count'][0],
                         start=-0.01, end=1.01, title=None):
        if stat == 'f1':
            stat_att = 'f1_perclass'
        elif stat == 'recall':
            stat_att = 'recall_perclass'
        elif stat == 'precision':
            stat_att = 'precision_perclass'

        true_count_perclass = self.best_epochs[0].count_perclass
        f1s_perclass = {'{} [{:>4}]'.format(c, true_count_perclass[i]): \
                            [getattr(e, stat_att)[i] for e in self.best_epochs] \
                        for i, c in enumerate(self.classes)}

        if sort_by == 'count':
            sort_by = sorted(self.classes, key=lambda c: self.best_epochs[0].count_perclass[self.classes.index(c)])

        if isinstance(sort_by, list):
            [f1s_perclass.pop(c) for c in list(f1s_perclass.keys()) if c.split(' ', 1)[0] not in sort_by]
            order = sorted(f1s_perclass, key=lambda c: sort_by.index(c.split(' ', 1)[0]))
        else:
            order = sorted(f1s_perclass, key=lambda c: sort_by(f1s_perclass[c]))

        df = pd.DataFrame(f1s_perclass, columns=order)
        ax = df.plot(kind='box', vert=False, figsize=(16, len(order)/5))
        ax.set_xlabel('{} Score'.format(stat))
        ax.grid(True, which='both', axis='x')
        ax.grid(True, which='major', axis='y')
        ax.set_xlim(start, end)
        if title is None:
            title = '{} Scores per Class for {} ({}x runs)'.format(stat, self.name, len(self.best_epochs))
        ax.set_title(title)
        return ax

    def combo_epoch(self):
        return ComboEpoch(self.best_epochs)

    def plot_loss_per_run(self):
        raise NotImplementedError


class Collection:
    """ A collection of Series of equal length and the same classes"""

    def __init__(self, series_list, root='output', best_epochs_only=False):
        self.srcs = [os.path.join(root,series) for series in series_list]
        self.collection = []
        for src in self.srcs:
            series = Series(src, best_epochs_only=best_epochs_only)
            if self.collection:
                assert len(series) == len(self.collection[-1])
                assert series.classes == self.collection[-1].classes
            self.collection.append( series )

    def boxplot(self, stat, start=None, end=1.01):
        stat_dict = {series.name: [getattr(be, stat) for be in series.best_epochs] for series in self.collection}

        df = pd.DataFrame(stat_dict)
        ax = df.plot(kind='box', vert=False, figsize=[12, len(self.collection)/2])
        ax.set_title('{} per Collection of Series'.format(stat))
        ax.grid(True, which='both', axis='x')
        ax.set_xlim(start, end)
        return ax









