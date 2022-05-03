import io
import os

import numpy as np
import argparse

import torch.onnx
from pytorch_lightning import seed_everything
from neuston_models import NeustonModel
from neuston_data import ImageDataset
from scipy.special import softmax
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def do_export(args):

    # load model
    classifier = NeustonModel.load_from_checkpoint(args.MODEL)
    classes = classifier.hparams.classes
    seed_everything(classifier.hparams.seed)
    classifier.eval()
    classifier.to(args.device)
    if args.half: classifier.half()
    classifier.freeze()
    if args.output:
        output = args.output
        os.makedirs(os.path.dirname(output), exist_ok=True)
    else:
        output = args.MODEL.replace('.ptl','.onnx')
        if args.batchsize: output = output.replace('.onnx',f'.B{args.batchsize}.onnx')
        if args.half: output = output.replace('.onnx','.FP16.onnx')

    print(str(type(classifier.model)))
    dummy_batch_size = args.batchsize or 10
    if 'inception' in str(type(classifier.model)):
        dummy_input = torch.randn(dummy_batch_size, 3, 299, 299, device=args.device)
    else:
        dummy_input = torch.randn(dummy_batch_size, 3, 224, 224, device=args.device)
    if args.half: dummy_input = dummy_input.half()

    # perform export
    torch.onnx.export(classifier.model,          # model being run
                      dummy_input,               # model input (or a tuple for multiple inputs)
                      output,                    # where to save the model (can be a file or file-like object)
                      export_params = True,      # store the trained parameter weights inside the model file
                      opset_version = args.opset,# the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes = None if args.batchsize else {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      #verbose=True,
                      )
    print('EXPORTED:',output)

    # include classes file
    output_classes = output.replace('.onnx','.classes')
    with open(output_classes,'w') as f:
        f.write('\n'.join(classes))
    print('EXPORTED:', output_classes)


def do_run(args):
    import onnxruntime as ort

    # inputs to array
    img_paths = []
    if os.path.isdir(args.SRC):
        for pardir, _, imgs in os.walk(args.SRC):
            imgs = [os.path.join(pardir, img) for img in imgs if img.endswith(IMG_EXTENSIONS)]
            img_paths.extend(imgs)
    elif os.path.isfile(args.SRC) and args.SRC.endswith(('.txt','.list')):  # TODO TEST: textfile img run
        with open(args.SRC, 'r') as f:
            img_paths = f.readlines()
            img_paths = [img.strip() for img in img_paths]
            img_paths = [img for img in img_paths if img.endswith(IMG_EXTENSIONS)]
    elif args.SRC.endswith(IMG_EXTENSIONS):  # single img # TODO TEST: single img run
        img_paths.append(args.SRC)
    image_dataset = ImageDataset(img_paths, resize=299, input_src=args.SRC)
    input_images = [path for _,path in image_dataset]
    input_array = np.asarray([img.numpy() for img,_ in image_dataset])
    #print(image_dataset)

    # do inference
    ort_session = ort.InferenceSession(args.MODEL)
    outputs = ort_session.run(None, {'input':input_array})
    out = np.asarray(outputs[0])
    out = softmax(out,axis=1)
    output_classes = np.argmax(out,axis=1)
    output_scores = np.max(out,axis=1)

    print(output_scores)
    print(output_classes)

    # get labels
    classfile = args.classfile or args.MODEL.replace('.onnx','.classes')
    print(classfile)
    if os.path.isfile(classfile):
        with open(classfile) as f:
            classes = [c.rstrip('\n') for c in f.readlines()]
        output_labels = [classes[idx] for idx in output_classes]
        print(output_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ptl models to ONNX')

    # Create subparsers
    subparsers = parser.add_subparsers(dest='cmd_mode', help='These sub-commands are mutually exclusive.')
    export = subparsers.add_parser('EXPORT', help='Export a .ptl model to .onnx')
    run = subparsers.add_parser('RUN', help='Run an onnx model')

    # EXPORT from .ptl
    export.add_argument('MODEL', help='Model .ptl file to convert')
    export.add_argument('--half', action='store_true', help='Exports model using 16bit floating point precision')
    export.add_argument('--device', default='cpu', choices=('cpu','cuda'), help='Device to load model and tensors to. Default is "cpu"')
    export.add_argument('--opset', default=12, type=int, help='Opset Version for onnx. Default is 12.')
    export.add_argument('--output', default=None, help='Same as model file but with ".ptl" replaced with ".onnx"')
    export.add_argument('--batchsize', type=int, help='Set a fixed batch size for the model')

    # RUN onnx
    run.add_argument('MODEL', help='onnx model file')
    run.add_argument('SRC', help='file to run the model on')
    run.add_argument('--classfile','-c', help='file with list of class labels')

    args = parser.parse_args()

    if args.cmd_mode=='EXPORT':
        do_export(args)
    else: # RUN
        do_run(args)

