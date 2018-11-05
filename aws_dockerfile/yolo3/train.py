#!/usr/bin/env python

from __future__ import print_function
import traceback
import os
import sys
import logging
import time
import warnings
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
from gluoncv.utils import LRScheduler
from gluoncv.data import LstDetection
import json

# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
training_channel = 'training'
validation_channel = 'validation'
synset_channel = 'synset'
training_path = os.path.join(input_path, training_channel)
validation_path = os.path.join(input_path, validation_channel)
synset_path = os.path.join(input_path, synset_channel)


class hyperparameters():
    def __init__(self):
        with open(param_path, 'r') as tc:
            hyperparameters = json.load(tc)
        self.data_shape = int(hyperparameters.get('data_shape', 416))
        self.batch_size = int(hyperparameters.get('batch_size', 32))
        self.num_workers = int(hyperparameters.get('num_workers', 0))
        self.epochs = int(hyperparameters.get('epochs', 200))
        self.lr = float(hyperparameters.get('learning_rate', 0.001))
        self.lr_decay = float(hyperparameters.get('learning_rate_decay', 0.1))
        self.lr_decay_epoch = hyperparameters.get('learning_rate_decay_epoch', '160,180')
        self.momentum = float(hyperparameters.get('momentum', 0.9))
        self.wd = float(hyperparameters.get('weight_decay', 0.0005))
        self.syncbn = hyperparameters.get('syncbn', True)
        self.start_epoch = int(hyperparameters.get('start_epoch', 0))
        self.log_interval = int(hyperparameters.get('log_interval', 100))
        self.save_interval = int(hyperparameters.get('save_interval', 10))
        self.export_epoch = int(hyperparameters.get('export_epoch', 200))
        self.val_interval = int(hyperparameters.get('val_interval', 10))
        self.resume = hyperparameters.get('resume', '')
        self.num_gpus = int(hyperparameters.get('num_gpus', 0))
        self.num_samples = int(hyperparameters.get('num_samples', -1))
        self.train_dataset = os.path.join(training_path, hyperparameters.get('train_dataset', 'train.lst'))
        self.valid_dataset = os.path.join(validation_path, hyperparameters.get('valid_dataset', 'valid.lst'))
        self.synset = os.path.join(synset_path, hyperparameters.get('synset', 'synset.txt'))
        self.save_prefix = hyperparameters.get('save_prefix', '')
        self.client_id = hyperparameters.get('client_id')
        # training contexts
        if self.num_gpus > 0:
            self.ctx = [mx.gpu(int(i)) for i in range(0, num_gpus)]
        else:
            self.ctx = [mx.cpu()]
        # network
        self.net_name = '_'.join(('yolo3', 'darknet53', 'custom'))
        gutils.random.seed(233)
    def get(self):
        return('data shape: ' + str(self.data_shape) +
               ' batch size: ' + str(self.batch_size) +
               ' epochs: ' + str(self.epochs) +
               ' number of workers: ' + str(self.num_workers) +
               ' learning rate: ' + str(self.lr) +
               ' learning rate decay: ' + str(self.lr_decay) +
               ' momentum: ' + str(self.momentum) +
               ' number of gpus: ' + str(self.num_gpus))



# The function to execute the training.
def train():
    try:
        hyparams = hyperparameters()
        hyparams.save_prefix = os.path.join(model_path, hyparams.client_id, hyparams.net_name)
        # use sync bn if specified
        num_sync_bn_devices = len(hyparams.ctx) if hyparams.syncbn else -1
        if num_sync_bn_devices > 1:
            classes = get_dataset_classes(hyparams.synset)
            net = get_model(hyparams.net_name, classes=classes, transfer='voc', pretrained_base=False, num_sync_bn_devices=num_sync_bn_devices)
            async_net = get_model(hyparams.net_name, classes=classes, pretrained_base=False, transfer='voc')  # used by cpu worker
        else:
            classes = get_dataset_classes(hyparams.synset)
            net = get_model(hyparams.net_name, classes=classes, transfer='voc', pretrained_base=True, pretrained=True) #, num_sync_bn_devices=num_sync_bn_devices)
            async_net = net
        if hyparams.resume.strip():
            net.load_parameters(hyparams.resume.strip())
            async_net.load_parameters(hyparams.resume.strip())
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                net.initialize()
                async_net.initialize()
        # training data
        train_dataset, val_dataset, eval_metric = get_dataset(hyparams.train_dataset, hyparams.valid_dataset, classes, hyparams)
        train_data, val_data = get_dataloader(
            async_net, train_dataset, val_dataset, hyparams.data_shape, hyparams.batch_size, hyparams.num_workers)

        # training
        train_yolo3(net, train_data, val_data, eval_metric, hyparams.ctx, hyparams)
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


def get_dataset(train_dataset_lst_file, valid_dataset_lst_file, classes, hyparams):
    train_dataset = LstDetection(train_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)
    valid_dataset = LstDetection(valid_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)
    val_metric = VOCMApMetric(iou_thresh=0.5, class_names=classes)
    if hyparams.num_samples < 0:
        hyparams.num_samples = len(train_dataset)
    return train_dataset, valid_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, True, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def train_yolo3(net, train_data, val_data, eval_metric, ctx, hyparams):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    # learning rate scheduler
    lr_scheduler = LRScheduler(mode='step',
                               baselr=hyparams.lr,
                               niters=hyparams.num_samples // hyparams.batch_size, # ca 140/8 = 16
                               nepochs=hyparams.epochs,
                               step=[int(i) for i in hyparams.lr_decay_epoch.split(',')],
                               step_factor=float(hyparams.lr_decay),
                               warmup_epochs= max(2, 1000 // (hyparams.num_samples // hyparams.batch_size)),
                               warmup_mode='linear')
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': hyparams.wd, 'momentum': hyparams.momentum, 'lr_scheduler': lr_scheduler},
        kvstore='local')

    # targets
    sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    l1_loss = gluon.loss.L1Loss()

    # metrics
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = hyparams.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(hyparams.get())
    logger.info('Start training from [Epoch {}]'.format(hyparams.start_epoch))
    best_map = [0]
    for epoch in range(hyparams.start_epoch, hyparams.epochs):
        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            lr_scheduler.update(i, epoch)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1) #trainer.step(batch_size)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            if hyparams.log_interval and not (i + 1) % hyparams.log_interval:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
            btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        if not (epoch + 1) % hyparams.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, hyparams.save_interval, hyparams.save_prefix)
        if epoch == hyparams.export_epoch:
            net.export(os.path.join(hyparams.save_prefix, hyparams.net_name) , epoch=hyparams.export_epoch)

def get_dataset_classes(synset_path):
    classes = []
    with open(synset_path) as syn:
        for line in syn:
            classes.append(line)
    return classes



if __name__ == '__main__':
    train()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
