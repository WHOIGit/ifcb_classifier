"""a module for defining model architecture"""

# built in imports
import os
import argparse

# 3rd party imports
import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.models as MODEL_MODULE
import pytorch_lightning as ptl
from sklearn import metrics
import numpy as np

# project imports #
from neuston_callbacks import save_class_scores

def get_namebrand_model(model_name, num_o_classes, pretrained=False):
    if model_name == 'inception_v3':
        model = MODEL_MODULE.inception_v3(pretrained)  #, num_classes=num_o_classes, aux_logits=False)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_o_classes)
        model.fc = nn.Linear(model.fc.in_features, num_o_classes)
    elif model_name == 'alexnet':
        model = getattr(MODEL_MODULE, model_name)(pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_o_classes)
    elif model_name == 'squeezenet':
        model = getattr(MODEL_MODULE, model_name+'1_1')(pretrained)
        model.classifier[1] = nn.Conv2d(512, num_o_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_o_classes
    elif model_name.startswith('vgg'):
        model = getattr(MODEL_MODULE, model_name)(pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_o_classes)
    elif model_name.startswith('resnet'):
        model = getattr(MODEL_MODULE, model_name)(pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_o_classes)
    elif model_name.startswith('densenet'):
        model = getattr(MODEL_MODULE, model_name)(pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_o_classes)
    else:
        raise KeyError("model unknown!")
    return model


class NeustonModel(ptl.LightningModule):
    def __init__(self, args):
        super().__init__()

        if isinstance(args,dict):
            args = argparse.Namespace(**args)
        self.args = args
        self.hparams = args
        self.criterion = nn.CrossEntropyLoss()
        self.model = get_namebrand_model(args.MODEL, len(args.classes), args.pretrained)

        self.train_loss = None
        self.best_val_loss = np.inf
        self.best_epoch = None
        self.validation_outputs = None
        self.validation_inputs = None
        self.validation_input_srcs = None

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def loss(self, inputs, outputs):
        if isinstance(outputs,tuple) and len(outputs)==2: # inception_v3
            outputs, aux_outputs = outputs
            loss1 = self.criterion(outputs, inputs)
            loss2 = self.criterion(aux_outputs, inputs)
            batch_loss = loss1+0.4*loss2
        else:
            batch_loss = self.criterion(outputs, inputs)
        return batch_loss

    # TRAINING #
    def training_step(self, batch, batch_nb):
        input_data, input_classes, input_src =  batch
        outputs = self.forward(input_data)
        batch_loss = self.loss(input_classes, outputs)
        output_classes = torch.argmax(outputs.logits,dim=1)
        f1_w = metrics.f1_score(input_classes.cpu(), output_classes.cpu(), average='weighted')
        return dict(loss=batch_loss, progress_bar={'f1_w':100*f1_w, 'best_ep':self.best_epoch})

    def training_epoch_end(self, steps):
        epoch_loss = torch.stack([batch['loss'] for batch in steps]).sum()
        self.train_loss = epoch_loss
        return dict(loss=epoch_loss)

    # Validation #
    def validation_step(self, batch, batch_idx):
        input_data, input_classes, input_src = batch
        outputs = self.forward(input_data)
        batch_loss = self.loss(input_classes, outputs)
        return dict(loss=batch_loss, outputs=outputs.cpu(), input_classes=input_classes.cpu(), input_srcs=input_src)

    def validation_epoch_end(self, steps):
        validation_loss = torch.stack([batch['loss'] for batch in steps]).sum()
        if validation_loss<self.best_val_loss:
            self.best_val_loss = validation_loss
            self.best_epoch = self.current_epoch

        self.validation_outputs =  nn.functional.softmax(torch.cat([batch['outputs'] for batch in steps],dim=0),dim=1).numpy()
        self.validation_inputs = input_classes = torch.cat([batch['input_classes'] for batch in steps],dim=0).numpy()
        self.validation_input_srcs = [item for sublist in [batch['input_srcs'] for batch in steps] for item in sublist]  # flatten list
        output_classes = np.argmax(self.validation_outputs, axis=1)

        f1_weighted = metrics.f1_score(input_classes, output_classes, average='weighted')
        f1_macro = metrics.f1_score(input_classes, output_classes, average='macro')

        log = dict(epoch=self.current_epoch, best= self.best_epoch==self.current_epoch,
                   train_loss=self.train_loss, val_loss=validation_loss,
                   f1_macro=f1_macro, f1_weighted=f1_weighted)

        #input_labels = [self.args.classes[c] for c in input_classes]
        #output_labels = [self.args.classes[c] for c in output_classes]
        #f1_perclass = metrics.f1_score(input_labels, output_labels, labels=self.args.classes, average=None)
        #recall_perclass = metrics.recall_score(input_labels, output_labels, labels=self.args.classes, average=None)
        #for i,c in enumerate(self.args.classes):
        #    log['f1_'+c] = f1_perclass[i]

        # todo save_class_scores()
        return dict(val_loss=validation_loss, log=log)

    # RUNNING the model #
    def test_step(self, batch, batch_idx):
        inputs,pids = batch
        outputs = self.forward(inputs)
        return dict(outputs=outputs.cpu(), pids=pids)

    def test_epoch_end(self, steps):

        outputs = torch.cat([batch['outputs'] for batch in steps],dim=0)
        outputs = nn.functional.softmax(outputs,dim=1).numpy()
        pids = [batch['pids'] for batch in steps]
        pids = [item for sublist in pids for item in sublist]  # flatten list

        bin_id = self.test_dataloader().dataset.bin.pid
        os.makedirs(self.args.run_outdir,exist_ok=True)
        outfile = os.path.join(self.args.run_outdir, self.args.run_outfile.format(bin=bin_id))
        save_class_scores(outfile, bin_id, outputs, pids, self.args.classes)
