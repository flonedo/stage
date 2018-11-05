"""Train SSD"""
import argparse
import os
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.accuracy import Accuracy
from customdetection import CustomDetection



# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def train(hyperparameters,
          input_data_config,
          channel_input_dirs,
          output_data_dir,
          model_dir,
          num_gpus,
          num_cpus,
          hosts,
          current_host,
          **kwargs):
    data_shape = hyperparameters.get('data_shape', 416)
    batch_size = hyperparameters.get('batch_size', 32)
    num_workers = hyperparameters.get('num_workers', 0)
    epochs = hyperparameters.get('epochs', 200)
    lr = hyperparameters.get('learning_rate', 0.0008)
    lr_decay = hyperparameters.get('learning_rate_decay', 0.1)
    lr_decay_epoch = hyperparameters.get('learning_rate_decay_epoch', '160,180')
    momentum = hyperparameters.get('momentum', 0.9)
    wd = hyperparameters.get('weight_decay', 0.0005)
    syncbn = hyperparameters.get('syncbn', True)
    start_epoch = hyperparameters.get('start_epoch', 0)
    log_interval = hyperparameters.get('log_interval', 100)
    save_interval = hyperparameters.get('save_interval', 10)
    export_epoch = hyperparameters.get('export_epoch', 200)
    val_interval = hyperparameters.get('val_interval', 10)
    resume = hyperparameters.get('resume', '')

    ctx = [mx.cpu()]
    training_dir = channel_input_dirs['training']

    # network
    classes = get_dataset_classes(synset)
    net_name = '_'.join(('ssd', str(300), 'vgg16_atrous', 'voc'))
    save_prefix = net_name
    net = get_model(net_name, pretrained_base=True)
    if resume.strip():
        net.load_parameters(resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()

    # training data


    train_dataset, val_dataset, eval_metric = get_dataset(args.train_dataset, args.valid_dataset, classes, args)
    train_data, val_data = get_dataloader(net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)

    # training
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': lr, 'wd': wd, 'momentum': momentum})

    # lr decay policy
    lr_decay = float(lr_decay)
    lr_steps = sorted([float(ls) for ls in lr_decay_epoch.split(',') if ls.strip()])

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    #logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(start_epoch, epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            if log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2))
        if (epoch % val_interval == 0) or (save_interval and epoch % save_interval == 0):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, save_interval, save_prefix)


def get_dataset(train_dataset_lst_file, valid_dataset_lst_file, classes, args):
    train_dataset = CustomDetection(train_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)
    valid_dataset = CustomDetection(valid_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    return train_dataset, valid_dataset, val_metric

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors=anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
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
        eval_metric.update([det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults])
    return eval_metric.get()

def get_dataset_classes(synset_path):
    classes = []
    with open(synset_path) as syn:
        for line in syn:
            classes.append(line)
    return classes