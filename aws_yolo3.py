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
from gluoncv.data.transforms.presets.yolo import load_test
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils import LRScheduler
from yolo3.customdetection import CustomDetection
from PIL import Image
import os
import json


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
    lr = hyperparameters.get('learning_rate', 0.001)
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

    gutils.random.seed(233)

    # training contexts
    if num_gpus > 0:
        ctx = [mx.gpu(int(i)) for i in range(0, num_gpus)]
    else:
        ctx = mx.cpu()

    # network
    net_name = '_'.join(('yolo3', 'darknet53', 'custom'))
    save_prefix = output_data_dir
    training_dir = channel_input_dirs['training']


    # use sync bn if specified
    num_sync_bn_devices = len(ctx) if syncbn else -1
    if num_sync_bn_devices > 1:
        classes = get_dataset_classes(os.path.join(training_dir, 'synset.txt'))
        net = get_model(net_name, classes=classes, transfer='voc', pretrained_base=False, num_sync_bn_devices=num_sync_bn_devices)
        async_net = get_model(net_name, classes=classes, pretrained_base=False, transfer='voc')  # used by cpu worker
    else:
        classes = get_dataset_classes(os.path.join(training_dir, 'synset.txt'))
        net = get_model(net_name, classes=classes, transfer='voc', pretrained_base=True, pretrained=True, num_sync_bn_devices=num_sync_bn_devices)
        async_net = get_model(net_name, classes=classes, pretrained_base=False, transfer='voc')
    if resume.strip():
        net.load_parameters(resume.strip())
        async_net.load_parameters(resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()


    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(os.path.join(training_dir, 'train', 'train.lst'), os.path.join(training_dir, 'valid', 'valid.lst'), classes)
    train_data, val_data = get_dataloader(async_net, train_dataset, val_dataset, data_shape, batch_size, num_workers)
    num_samples = len(train_dataset)
    # training
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    # learning rate scheduler
    lr_scheduler = LRScheduler(mode='step',
                               baselr=lr,
                               niters=num_samples // batch_size,
                               nepochs=epochs,
                               step=[int(i) for i in lr_decay_epoch.split(',')],
                               step_factor=float(lr_decay),
                               warmup_epochs= max(2, 1000 // (num_samples // batch_size)),
                               warmup_mode='linear')
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': wd, 'momentum': momentum, 'lr_scheduler': lr_scheduler},
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
    log_file_path = os.path.join(output_data_dir, net_name + '_train.log')
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    #logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(start_epoch))
    best_map = [0]
    for epoch in range(start_epoch, epochs):
        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=[ctx], batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=[ctx], batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=[ctx], batch_axis=0)
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
            if log_interval and not (i + 1) % log_interval:
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
        if not (epoch + 1) % val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, save_interval, save_prefix, net_name)
        if epoch == export_epoch:
            net.export(os.path.join(output_data_dir, "yolo"), epoch=export_epoch)


def get_dataset(train_dataset_lst_file, valid_dataset_lst_file, classes):
    train_dataset = CustomDetection(train_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)
    valid_dataset = CustomDetection(valid_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
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


def save_params(net, best_map, current_map, epoch, save_interval, prefix, net_name):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(os.path.join(prefix, ney_name), epoch, current_map))
        with open(os.path.join(prefix, net_name + '_best_map.log'), 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(os.path.join(prefix, ney_name), epoch, current_map))


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=[ctx], batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=[ctx], batch_axis=0, even_split=False)
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


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #


def model_fn(model_dir):
    net = gluon.nn.SymbolBlock.imports(os.path.join(model_dir, 'yolo3_darknet53_custom-symbol.json'), ['data'], os.path.join(model_dir, 'yolo3_darknet53_custom.params')
    return net

def transform_fn(model, input_data, content_type, accept):
    classes = model.classes()
    parsed = json.loads(input_data)
    outputs = []
    for row in parsed:
        x, img = load_test(row, short=416)
        # detects
        class_IDs, scores, bounding_boxes = model(x)
        image = Image.open(row)
        original_width, original_height = image.size
        scale = get_bounding_box_scale(416, original_width, original_height)
        # it shows the best 5 results
        detections = []
        for i in range(0, 4):
            xmin = scale*((bounding_boxes[0][i][0]).asscalar())
            ymin = scale*((bounding_boxes[0][i][1]).asscalar())
            xmax = scale*((bounding_boxes[0][i][2]).asscalar())
            ymax = scale*((bounding_boxes[0][i][3]).asscalar())
            score = float((scores[0][i]).asscalar())
            id = int((class_IDs[0][i]).asscalar())
            cl = classes[int(id)]
            perc_bounding_box = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            detections.append({'class': cl, 'score': score, 'boundingBoxPercentage': perc_bounding_box})
        outputs.append({'uri': row, 'detections': detections})
    return json.dumps(outputs), accept



def get_bounding_box_scale(resize, width, height):
    shortest = min(width, height)
    return 1.0/(float(resize)/float(shortest))


# ------------------------------------------------------------ #
# Utils                                                        #
# ------------------------------------------------------------ #

# reads classes from a text file
def get_dataset_classes(synset_path):
    classes = []
    with open(synset_path) as syn:
        for line in syn:
            classes.append(line)
    return classes