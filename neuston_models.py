"""a module for defining model architecture"""

# built in imports
import argparse

# 3rd party imports
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.functional import softmax
import torchvision.models as MODEL_MODULE
from torchvision.models.inception import InceptionOutputs
import pytorch_lightning as ptl
from sklearn import metrics
import numpy as np
import ifcb

# project imports #
from neuston_data import IfcbBinDataset
from inception_v4 import inceptionv4

def get_namebrand_model(model_name, num_o_classes, pretrained=False, metadata=False):
    if metadata:
        model = MetadataModel(model_name, num_o_classes, pretrained)
    elif model_name == 'inception_v3':
        model = MODEL_MODULE.inception_v3(pretrained)  #, num_classes=num_o_classes, aux_logits=False)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_o_classes)
        model.fc = nn.Linear(model.fc.in_features, num_o_classes)
        # TODO monkey wrench an adl. input layer towards the end for metadata injection
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
    elif model_name == 'inception_v4':
        if pretrained:
            model = inceptionv4(num_classes=1001, pretrained='imagenet+background')
            #model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_o_classes)
            model.last_linear = nn.Linear(model.last_linear.in_features, num_o_classes)
        else:
            model = inceptionv4(num_classes=num_o_classes, pretrained=None)
    else:
        raise KeyError("model unknown!", model_name)
    return model



class MetadataModel(nn.Module):
    def __init__(self, model_name, num_o_classes, pretrained, metadata_dim=2):  #metadata_layer_nodes=('N',),
        super(MetadataModel, self).__init__()
        self.cnn = get_namebrand_model(model_name, num_o_classes, pretrained)

        hidden_layer = num_o_classes
        self.fc1 = nn.Linear(num_o_classes + metadata_dim, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_o_classes)

    def forward(self, inputs):
        image,metadata = inputs
        cnn_outputs = self.cnn(image)
        if isinstance(cnn_outputs, tuple) and len(cnn_outputs) == 2:  # inception_v3
            cnn_outputs, aux_outputs = cnn_outputs

        x = torch.cat((cnn_outputs, metadata), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NeustonModel(ptl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams,dict):
            hparams = argparse.Namespace(**hparams)
        try:
            if not hparams.metadata_enable:
                del hparams.metadata_scaling
                del hparams.metadata_options
        except AttributeError: pass
        self.save_hyperparameters(hparams)

        self.criterion = nn.CrossEntropyLoss()

        if 'metadata_enable' in hparams and hparams.metadata_enable:
            self.model = MetadataModel(hparams.MODEL, len(hparams.classes), hparams.pretrained)
        else:
            self.model = get_namebrand_model(hparams.MODEL, len(hparams.classes), hparams.pretrained)

        # Instance Variables
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.agg_train_loss = 0.0

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        outputs = self.model(inputs)
        # inputs is a tuple of img_data,metadata for MetadataModel models
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
        input_data, input_classes, input_src = batch
        outputs = self.forward(input_data)
        batch_loss = self.loss(input_classes, outputs)
        self.agg_train_loss += batch_loss.item()
        return dict(loss=batch_loss)

    def training_epoch_end(self, steps):
        train_loss = torch.stack([batch['loss'] for batch in steps]).sum().item()
        #print('training_epoch_end: self.agg_train_loss={:.5f}, train_loss={:.5f}, DIFF={:.9f}'.format(self.agg_train_loss, train_loss, self.agg_train_loss-train_loss), end='\n\n')
        #return dict(train_loss=train_loss)

    # Validation #
    def validation_step(self, batch, batch_idx):
        input_data, input_classes, input_src = batch
        outputs = self.forward(input_data)
        val_batch_loss = self.loss(input_classes, outputs)
        outputs = outputs.logits if isinstance(outputs,InceptionOutputs) else outputs
        outputs = softmax(outputs,dim=1)
        return dict(val_batch_loss=val_batch_loss,
                    val_outputs=outputs,
                    val_input_classes=input_classes,
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

        outputs = torch.cat([batch['val_outputs'] for batch in steps],dim=0).detach().cpu().numpy()
        output_classes = np.argmax(outputs, axis=1)
        input_classes = torch.cat([batch['val_input_classes'] for batch in steps],dim=0).detach().cpu().numpy()
        input_srcs = [item for sublist in [batch['val_input_srcs'] for batch in steps] for item in sublist]

        f1_weighted = metrics.f1_score(input_classes, output_classes, average='weighted')
        f1_macro = metrics.f1_score(input_classes, output_classes, average='macro')

        eoe = 'Best Epoch: {}, train_loss: {:.3f}, val_loss: {:.3f}, val_f1_w={:02.1f}%, val_f1_m={:02.1f}%'
        eoe = eoe.format(True if self.current_epoch==self.best_epoch else self.best_epoch+1, self.agg_train_loss, validation_loss, 100*f1_weighted, 100*f1_macro)
        print(eoe, flush=True, end='\n\n')  # so slurm output can be followed along

        # used by callbacks and logger
        self.log('epoch', self.current_epoch, on_epoch=True)
        self.log('best', self.best_epoch==self.current_epoch, on_epoch=True)
        self.log('train_loss', self.agg_train_loss, on_epoch=True)
        self.log('val_loss', validation_loss, on_epoch=True)

        # csv_logger logger hacked to not include these in epochs.csv output
        self.log('input_classes', input_classes, on_epoch=True)
        self.log('output_classes', output_classes, on_epoch=True)
        self.log('input_srcs', input_srcs, on_epoch=True)
        self.log('outputs', outputs, on_epoch=True)

        # these will apppear in epochs.csv, but are not used by callbacks
        self.log('f1_macro',f1_macro, on_epoch=True)
        self.log('f1_weighted',f1_weighted, on_epoch=True)

        # Cleanup
        self.agg_train_loss = 0.0

        return dict(hiddens=dict(outputs=outputs))

    # RUNNING the model #
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        input_data, input_srcs = batch
        outputs = self.forward(input_data)
        outputs = outputs.logits if isinstance(outputs,InceptionOutputs) else outputs
        outputs = softmax(outputs, dim=1)
        return dict(test_outputs=outputs, test_srcs=input_srcs)

    def test_epoch_end(self, steps):

        # handle single and multiple test dataloaders
        datasets = self.test_dataloader()
        if isinstance(datasets, list): datasets = [ds.dataset for ds in datasets]
        else: datasets = [datasets.dataset]
        if isinstance(steps[0],dict):
            steps = [steps]

        RRs = []
        for steps,dataset in zip(steps,datasets):
            outputs = torch.cat([batch['test_outputs'] for batch in steps],dim=0).detach().cpu().numpy()
            images = [batch['test_srcs'] for batch in steps]
            images = [item for sublist in images for item in sublist]  # flatten list
            if isinstance(dataset, IfcbBinDataset):
                input_obj = dataset.bin.pid
            else:
                input_obj = dataset.input_src  # a path string
            rr = self.RunResults(inputs=images, outputs=outputs, input_obj=input_obj)
            RRs.append(rr)
        self.log('RunResults',RRs)
        #return dict(RunResults=RRs)

    class RunResults:
        def __init__(self, inputs, outputs, input_obj):
            self.inputs = inputs
            self.outputs = outputs
            self.input_obj = input_obj
            self.type = 'Bin' if isinstance(input_obj,ifcb.Pid) else 'ImgDir'
        def __repr__(self):
            rep = '{}: {} ({} imgs)'.format(self.type, self.input_obj, len(self.inputs))
            return repr(rep)
