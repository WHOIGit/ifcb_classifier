"""a module for defining model architecture"""

# built in imports
import argparse

# 3rd party imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import softmax
import torchvision.models as MODEL_MODULE
from torchvision.models.inception import InceptionOutputs
import pytorch_lightning as ptl
from sklearn import metrics
import numpy as np

# project imports #
from neuston_data import IfcbBinDataset


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
    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams,dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.criterion = nn.CrossEntropyLoss()
        self.model = get_namebrand_model(hparams.MODEL, len(hparams.classes), hparams.pretrained)

        # Instance Variables
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.agg_train_loss = 0.0

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
        self.agg_train_loss += batch_loss.item()
        return dict(loss=batch_loss)

    def training_epoch_end(self, steps):
        train_loss = torch.stack([batch['loss'] for batch in steps]).sum().item()
        #print('training_epoch_end: self.agg_train_loss={:.5f}, train_loss={:.5f}, DIFF={:.9f}'.format(self.agg_train_loss, train_loss, self.agg_train_loss-train_loss), end='\n\n')
        self.agg_train_loss = 0.0
        return dict(train_loss=train_loss)

    # Validation #
    def validation_step(self, batch, batch_idx):
        input_data, input_classes, input_src = batch
        outputs = self.forward(input_data)
        val_batch_loss = self.loss(input_classes, outputs)
        outputs = outputs.logits if isinstance(outputs,InceptionOutputs) else outputs
        outputs = softmax(outputs,dim=1)
        return dict(val_batch_loss=val_batch_loss.cpu(),
                    val_outputs=outputs.cpu(),
                    val_input_classes=input_classes.cpu(),
                    val_input_srcs=input_src)

    def validation_epoch_end(self, steps):
        print(end='\n\n') # give space for progress bar
        if self.current_epoch==0: self.best_val_loss = np.inf  # takes care of any lingering val_loss from sanity checks

        validation_loss = torch.stack([batch['val_batch_loss'] for batch in steps]).sum()
        #eoe0 = 'validation_epoch_end: best_val_loss={}, curr_val_loss={}, curr<best={}, curr-best (neg is good)={}'
        #eoe0 = eoe0.format(self.best_val_loss, validation_loss.item(), validation_loss.item()<self.best_val_loss, validation_loss.item()-self.best_val_loss)
        #print(eoe0)

        if validation_loss.item()<self.best_val_loss:
            self.best_val_loss = validation_loss.item()
            self.best_epoch = self.current_epoch

        outputs = torch.cat([batch['val_outputs'] for batch in steps],dim=0).numpy()
        output_classes = np.argmax(outputs, axis=1)
        input_classes = torch.cat([batch['val_input_classes'] for batch in steps],dim=0).numpy()
        input_srcs = [item for sublist in [batch['val_input_srcs'] for batch in steps] for item in sublist]

        f1_weighted = metrics.f1_score(input_classes, output_classes, average='weighted')
        f1_macro = metrics.f1_score(input_classes, output_classes, average='macro')

        log = dict(epoch=self.current_epoch, best = self.best_epoch==self.current_epoch,
                   train_loss=self.agg_train_loss, val_loss=validation_loss,
                   f1_macro=f1_macro, f1_weighted=f1_weighted)

        eoe = 'Best Epoch: {}, train_loss: {:.3f}, val_loss: {:.3f}, val_f1_w={:02.1f}%, val_f1_m={:02.1f}%'
        eoe = eoe.format(True if self.current_epoch==self.best_epoch else self.best_epoch+1, self.agg_train_loss, validation_loss, 100*f1_weighted, 100*f1_macro)
        print(eoe, flush=True, end='\n\n')  # so slurm output can be followed along

        return dict(val_loss=validation_loss, log=log,
                    input_classes=input_classes, output_classes=output_classes,
                    input_srcs=input_srcs, outputs=outputs)

    # RUNNING the model #
    def test_step(self, batch, batch_idx):
        input_data, input_srcs = batch
        outputs = self.forward(input_data)
        outputs = outputs.logits if isinstance(outputs,InceptionOutputs) else outputs
        outputs = softmax(outputs, dim=1)
        return dict(test_outputs=outputs.cpu(), test_srcs=input_srcs)

    def test_epoch_end(self, steps):
        outputs = torch.cat([batch['test_outputs'] for batch in steps],dim=0).numpy()
        images = [batch['test_srcs'] for batch in steps]
        images = [item for sublist in images for item in sublist]  # flatten list
        dataset = self.test_dataloader().dataset
        if isinstance(dataset, IfcbBinDataset):
            bin_id = str(dataset.bin.pid)
        else: bin_id = 'NaB'
        rr = self.RunResults(inputs=images, outputs=outputs, bin_id=bin_id)
        return dict(RunResults=rr)

    class RunResults:
        def __init__(self, inputs, outputs, bin_id):
            self.inputs = inputs
            self.outputs = outputs
            self.bin_id = bin_id
        def __repr__(self):
            rep = 'Bin: {} ({} imgs)'.format(self.bin_id, len(self.inputs))
            return repr(rep)
