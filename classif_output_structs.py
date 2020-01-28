import os
from ast import literal_eval
from sklearn import metrics
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from statistics import stdev, mean

import plotutil


class Epoch:
    """A Single epoch of a classification-training run"""
    def __init__(self,epoch_dict):
        """

        :param epoch_dict: epoch reults dict or filepath thereto. Must contain keys:
            true_inputs,true images, prediction_outputs, epoch, classes, eval_loss, training_loss,
            secs_elapsed, name, output_dir, model, training_dir, evaluation_dir, pretrained,no_normalize,
            min_epochs,max_epochs,batch_size,loaders,augment,learning_rate
        """

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
        try: self.prediction_ranks   = epoch_dict['prediction_ranks']
        except KeyError: self.prediction_ranks = NotImplementedError
        try: self.prediction_scores  = epoch_dict['prediction_allscores']
        except KeyError: self.prediction_scores = NotImplementedError 

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
        self.pretrained     = epoch_dict['pretrained']
        self.normalized     = not epoch_dict['no_normalize']
        self.min_epochs     = epoch_dict['min_epochs']
        self.max_epochs     = epoch_dict['max_epochs']
        self.batch_size     = epoch_dict['batch_size']
        self.loaders        = epoch_dict['loaders']
        self.augments       = epoch_dict['augment']
        self.learning_rate  = epoch_dict['learning_rate']
        try:
            self.training_dir   = epoch_dict['training_dir']
            self.eval_dir       = epoch_dict['evaluation_dir']
            self.src = (self.training_dir, self.eval_dir)
        except KeyError:
            self.src = epoch_dict['src'] if 'src' in epoch_dict else epoch_dict['root'] # todo delete this eventually
            self.seed = epoch_dict['seed']
            self.split = epoch_dict['split'].split(':')
            self.swap = epoch_dict['swap'] # typically either None or True
            if self.split == (50,50) and self.swap is None:
                self.swap = False
            elif self.swap:
                self.split = self.split[1],self.split[0]
            self.training_dir = dict(src=self.src, split='{}%'.format(self.split[0]), seed=self.seed, swap=self.swap)
            self.eval_dir     = dict(src=self.src, split='{}%'.format(self.split[1]), seed=self.seed, swap=self.swap)
            self.class_minimum = epoch_dict['class_minimum']
            self.class_config = epoch_dict['class_config']
        # derived values, non proc intensive. otherwise see __getattr__
        self.hours_elapsed = round(self.secs_elapsed/60/60, 2)
        self.mins_per_epoch = round(self.secs_elapsed/60/self.epoch_num, 2)

    @property
    def input_labels(self): return [self.classes[i] for i in self.true_inputs]

    @property
    def output_labels(self): return [self.classes[i] for i in self.prediction_outputs]

    def __getattr__(self, name: str):
        """Returns the attribute matching passed name if possible, else tries to calculate the value"""
        
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


    def get_input_metadata(self):
        """Returns dict of training-input parameters at the Run level: <as Series metedata, plus...> name,output_dir"""
        input_metadata = {}
        for a in Run.metadata_attribs:
            input_metadata[a] = self.__getattr__(a)
        return input_metadata
    def get_series_metadata(self):
        """Returns dict of training-input parameters: at the Series level; classes,model,root,pretrained,no_normalize,min_epochs,max_epochs,batch_size,loaders,augment,learning_rate"""
        series_metadata = {}
        for a in Series.metadata_attribs:
            series_metadata[a] = self.__getattr__(a)
        return series_metadata


    def plot_confusion_matrix(self, title=None, output='show', sort_by='recall'):
        """Plots a confusion matrix for all classes in epoch's results.

        :param title: The plot's title. Default is "{name} (epoch={}) f1_w={}% f1_m={}"
        :param output: How to output plot. Options are "show" or a string filepath to save the plot to. Default is "show"
        :param sort_by: How to sort the classes on both axis. Options are "recall" and "alphabetical"

        see:  plotutil.make_confusion_matrix_plot for more info
        """
        if title is None:
            title = '{} (epoch={}) f1_w={:.1f}% f1_m={:.1f}%'.format(self.name,self.epoch_num,100*self.f1_weighted,100*self.f1_macro)

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

    def perclass_barplot(self, stat='f1', start=None, end=None, sort_by=['max','min','alphabetical','count'][1]):
        if stat == 'f1': stat_att = 'f1_perclass'
        elif stat == 'recall': stat_att = 'recall_perclass'
        elif stat == 'precision': stat_att = 'precision_perclass'
        elif stat == 'count': stat_att = 'count_perclass'

        #TODO sorting
        #TODO stat = count

        stat_dict = {'classes':self.classes, stat:getattr(self,stat_att), 'counts':self.count_perclass}
        stat_dict['classes'] = ['{} [{:>4}]'.format(c, self.count_perclass[i]) for i,c in enumerate(self.classes)]
        df = pd.DataFrame(stat_dict)
        df = df.sort_values(by=stat)
                        
        ax = df.plot.barh(x='classes', y=stat, figsize=[12, len(self.classes)/3])
        ax.set_title('Perclass {} for {}'.format(stat,self.name))
        ax.grid(True, which='both', axis='x')
        ax.set_xlim(start,end)
        return ax


    def plot_confidence_histogram(self, good_only=None, bad_only=None, classes=None):
        if good_only is None and bad_only is None:
            good_only = bad_only = True
        assert good_only != False and bad_only != False

        if classes is None:
            title = 'Histogram of Classification Confidence'
            if good_only and bad_only:
                data = self.prediction_ranks
            elif good_only:
                data = [pr for pr,ti,po in zip(self.prediction_ranks,self.true_inputs,self.prediction_outputs) if ti==po]
                title += ' (Correctly Classified Only)'
            elif bad_only:
                data = [pr for pr,ti,po in zip(self.prediction_ranks,self.true_inputs,self.prediction_outputs) if ti!=po]
                title += ' (Incorrectly Classified Only)'

            df = pd.DataFrame(data)
            ax = df.T.max().hist(bins=20)
            ax.set(title=title, xlim=(0, 1))
            ax.set_xlabel('Confidence')
            ax.set_ylabel('image count')

        else:

            if good_only and bad_only:
                data = zip(self.prediction_ranks, self.output_labels)
            elif good_only:
                data = [(pr,ol) for pr,ol,il in zip(self.prediction_ranks,self.output_labels,self.input_labels) if ol==il]
            elif bad_only:
                data = [(pr,ol) for pr,ol,il in zip(self.prediction_ranks,self.output_labels,self.input_labels) if ol!=il]

            df = pd.DataFrame(data, columns=['confidence','class'])
            gb = df.groupby('class')
               # TODO ORDER results, also this is really hacky
            if classes == 'all':
                classes = self.classes[:]
            elif isinstance(classes,str):
                classes = [classes]

            fig, axes = plt.subplots(figsize=(8, 0.8*len(classes)), nrows=len(classes), ncols=1)
            if isinstance(axes,mpl.axes._axes.Axes):axes = [axes]
            #for group, ax in zip(gb, axes):
            #    name, group = group
            #    ax = group.hist(ax=ax, bins=20)[0]
            #    ax.set(xlim=(0, 1), title=None)
            #    ax.set_ylabel(name, rotation=0, ha='right', va='bottom')

            for i,ax in enumerate(axes):
                for group in gb:
                    name,group = group
                    if name in classes:
                        classes.remove(name)
                        ax = group.hist(ax=ax,bins=20)[0]
                        ax.set(xlim=(0, 1),title=None)
                        ax.set_ylabel(name, rotation=0, ha='right', va='bottom')
                        break
            plt.tight_layout()


    def pairwise_df(self, skip_correct=False, skip_empties=False ):


        # create a confusion matrix to count mis/classification instances
        cm = metrics.confusion_matrix(self.input_labels,self.output_labels,self.classes)

        # Create a stacked dataframe of the confusin matrix such that input classes, output classes,
        # and classification count are all separate columns
        df_matrix = pd.DataFrame(cm,index=self.classes,columns=self.classes)
        df_matrix.index.name = 'Input'
        df_matrix.columns.name = 'Output'
        df = df_matrix.stack().rename("count").reset_index()

        ## Adding images column using df.apply and an images per-input-output-class dict
        # ci=class input, cp=class predicted
        pairwise_metadata = {ci:{cp:{'images':[]} for cp in self.classes} for ci in self.classes}
        combo = zip(self.input_labels,self.true_images,self.output_labels)
        for input_label,input_image,output_label in combo:
            pairwise_metadata[input_label][output_label]['images'].append(input_image)

        def add_images(row):
            # function for df.apply, returns list of images on per-row ie per input-output class basis
            input_label = row[0]
            output_label = row[1]
            images = pairwise_metadata[input_label][output_label]['images']
            return images

        # creation of new images column
        df['images'] = df.apply(add_images, axis=1)

        if skip_correct:
            # exclude rows where input and output class are the same ie were correctly classified
            df = df[df['Input'] != df['Output']]
        if skip_empties:
            # exclude input-output classes that were never predicted
            df = df[df['count'] != 0]

        # set input and output columns to be the multi-index.
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
                e = Epoch(e)
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

        # gives us a df with input-output class-pair index and the columns: count, images
        df = super().pairwise_df(skip_correct=skip_correct, skip_empties=skip_empties)

        ## parse which images to keep based on number of times that image has been predicted for given class pair
        if keep_dupes is None: # keep all images
            keep_dupes = range(1,len(self.epochs)+1)
        elif isinstance(keep_dupes,int):
            keep_dupes = [keep_dupes]
        elif isinstance(keep_dupes,list): pass  # native behavior
        elif keep_dupes.startswith('>='):
            value = int(keep_dupes.replace('>=',''))
            keep_dupes = range(value,len(self.epochs)+1)
        elif keep_dupes.startswith('>'):
            value = int(keep_dupes.replace('>',''))
            keep_dupes = range(value+1,len(self.epochs)+1)
        elif '-' in keep_dupes:
            # range of values, eg "3-5" -> [3,4,5]
            v1,v2 = keep_dupes.split('-')
            v1,v2 = int(v1),int(v2)
            keep_dupes = range(v1,v2+1)
        else:
            # a string integer
            try: keep_dupes = [int(keep_dupes)]
            except: raise ValueError(keep_dupes,'not valid')
        keep_dupes = list(keep_dupes)

        def denote_duplicates(row):
        # function for df.apply to add image_counts duplicates and update imags and counts to not have/correctly count misclassifications
            # keep only unique images, and count how many duplicates there were of each image
            image_set = sorted(set(row['images']))
            image_set_counts = [len([i for i in row['images'] if i == img]) for img in image_set]

            # sort images by most repeated to least
            if image_set:  # doesn't work for empty lists
                # order images by most frequent first
                count_image_tups = sorted(zip(image_set_counts, image_set), reverse=True)
                # keep only images with a certain number or range of duplicates
                count_image_tups = [tup for tup in count_image_tups if tup[0] in keep_dupes]
                if len(count_image_tups) == 0: return [],[],0   # next lines don't work on empty lists
                # convert ordered+redued tuple-pair back to lists
                image_set_counts, image_set = list(zip(*count_image_tups))
                image_set_counts, image_set = list(image_set_counts), list(image_set)

            # retur new/updates row values
            return image_set, image_set_counts, len(image_set)

        df['images'], df['image_counts'], df['count'] = zip(*df.apply(denote_duplicates, axis=1))
        if skip_empties:
            # exclude input-output classes that were never predicted at the specified keep_dupes threshold
            df = df[df['count'] != 0]

        if naughty_sort:
            # sort rows my most duplicated and then by most frequently misclassified.
            df['naughty_rank'] = df.image_counts.apply(lambda counts: sum(c**2 for c in counts))
            df = df.sort_values('naughty_rank',ascending=False)
            df = df.drop('naughty_rank',1)

        return df

    def naughty_dupes(self,minimum=1,plot=False):
        """Creates df (or plat of df) containing histogram of chronically mis-classified images

        :param minimum: minimum numbe of duplicates to include. Default is 1
        :param plot: boolean. if True, return a plot axes, else returns a df. Default is False
        """
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
                       ['model', 'src',
                       #'training_dir', 'eval_dir',
                        'pretrained', 'normalized',
                        'augments', 'learning_rate',
                        'min_epochs', 'max_epochs',
                        'batch_size', 'loaders']

    def __init__(self, evaluation_records:str):
        self.src = evaluation_records

        self.epochs = []
        with open(evaluation_records) as f:
            epoch_dicts = literal_eval(f.read())
            for epoch_dict in epoch_dicts:
                self.epochs.append( Epoch(epoch_dict) )

        self.best_epoch_num = 0
        self.best_eval_loss = float('inf')
        for i,epoch in enumerate(self.epochs):
            if epoch.eval_loss < self.best_eval_loss:
                self.best_epoch_num = i
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

    def __init__(self, series_dir, best_epochs_only=False, ignore_metadata=False):
        self.src = series_dir
        self.name = os.path.basename(series_dir)

        runs_or_epochs = []
        subdirs = [os.path.join(series_dir, subdir) for subdir in os.listdir(series_dir)]
        for subdir in subdirs:
            if os.path.isdir(subdir):
                if best_epochs_only:
                    run_or_epoch = os.path.join(subdir, 'best_epoch.dict')
                    run_or_epoch = Epoch(run_or_epoch)
                else:
                    run_or_epoch = os.path.join(subdir,'evaluation_records.lod')
                    run_or_epoch = Run( run_or_epoch )
                if runs_or_epochs and not ignore_metadata:
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
        elif input_stats == 'f1':
            stats = 'f1_weighted f1_macro'.split()
        elif input_stats == 'recall':
            stats = 'recall_weighted recall_macro'.split()
        elif input_stats == 'precision':
            stats = 'precision_weighted precision_macro'.split()
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
        elif sort_by == 'mean':
            sort_by = mean
        elif sort_by in ['stdev','stddev']:
            sort_by = stdev

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

    def plot_stat_vs_perclass_count(self, stat='f1', figsize=(8,4), logx=True, plot_error=False):
        counts = self.best_best_epoch.count_perclass  # should be the same for all counts assuming run/epoch seeds all the same

        if stat in ['f1','recall','precision']: stat_pc=stat+'_perclass'
        else: stat_pc = stat

        stat_means = [mean([be.__getattr__(stat_pc)[i] for be in self.best_epochs]) for i, c in enumerate(self.classes)]
        errs = [stdev([be.__getattr__(stat_pc)[i] for be in self.best_epochs]) for i, c in enumerate(self.classes)]
        if plot_error:
            stat_means = errs
            errs = [0 for c in self.classes]
        df_xy = pd.DataFrame(dict(counts=counts, stat_means=stat_means, errs=errs, classes=self.classes))

        ax = df_xy.plot.scatter('counts', 'stat_means', yerr='errs', logx=logx,
                                title='Scatter plot of {} {}_scores vs image counts, per-class'.format('error-spread of' if plot_error else 'mean',stat),
                                figsize=figsize)
        ax.set_ylabel(stat+' stdev' if plot_error else stat+' mean')
        ax.grid(True, which='both', axis='x')

        # formatting xticks
        if logx:

            def logx_125(n):
                """generator that returns a sequence like [1,2,5,10,20,50,100,200,500,...] for log scale"""
                num = 1
                yield num
                while num < n:
                    msb = int(str(num)[0])
                    if msb == 1 or msb == 5:
                        num = 2*num
                    else:
                        num = int(2.5*num)  #msb == 2
                    yield num

            ticks = list(logx_125(max(counts)))
            ax.set_xlim(0.9, ticks[-1])
            ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())


    def best_epoch_combo(self):
        return ComboEpoch(self.best_epochs)

    @property
    def best_best_epoch(self):
        return sorted(self.best_epochs, key=lambda be: be.eval_loss)[0]

    def plot_loss_per_run(self):
        raise NotImplementedError


class Collection:
    """ A collection of Series of equal length and the same classes"""

    def __init__(self, series_list, root='output', best_epochs_only=False, same_classes=True, ignore_series_metadata=False):
        self.srcs = [os.path.join(root,series) for series in series_list]
        self.collection = []
        for src in self.srcs:
            series = Series(src, best_epochs_only=best_epochs_only, ignore_metadata=ignore_series_metadata)
            if self.collection:
                assert len(series) == len(self.collection[-1])
                if same_classes: assert series.classes == self.collection[-1].classes
            self.collection.append( series )

    def boxplot(self, stat, start=None, end=1.01):
        stat_dict = {series.name: [getattr(be, stat) for be in series.best_epochs] for series in self.collection}

        df = pd.DataFrame(stat_dict)
        ax = df.plot(kind='box', vert=False, figsize=[12, len(self.collection)/2])
        ax.set_title('{} per Collection of Series'.format(stat))
        ax.grid(True, which='both', axis='x')
        ax.set_xlim(start, end)
        return ax









