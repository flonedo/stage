from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
from mxnet import autograd
import os
from gluoncv.data.transforms import presets
from yolo3.customdetection import *


if __name__ == '__main__':
    # params
    batch_size = 4
    train_dataset_lst_file = 'train.lst'
    valid_dataset_lst_file = 'valid.lst'
    width = 416
    height = 416
    num_workers = 0
    ctx = mx.cpu()
    classes = ['Alfa Romeo']


    # lst file creation
    # create_lst(train_dataset_lst_file, jpgpath='JPGImages/', resize=True, width=416, height=416)
    # create_lst(valid_dataset_lst_file, jpgpath='JPGImages/', resize=True, width=416, height=416)

    # import datasets
    # flag=1 RBG image, flag=0 bw image
    train_dataset = CustomDetection(train_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)
    valid_dataset = CustomDetection(valid_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)

    # network
    net = model_zoo.get_model('yolo3_darknet53_custom', classes=classes, pretrained_base=False)
    net.initialize()
    # loss
    loss = gcv.loss.YOLOV3Loss()

    # split datasets into batches
    # batchify function: single images  from the dataset are stacked on each other; pad adds pad_vals to even different
    # length rows
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
    # shuffle=True loads images from the dataset in random order
    # last_batch sets what to do with image reminder in case dataset cannot be split exactly into batch_size batches.
    # last_batch=rollover puts remaining images in the next epoch
    # last_batch=discard doesn't use the incomplete batch
    # last_batch=keep uses the incomplete batch
    '''To work yolo network wants a specific input data format; such format is easily provided by YOLO3defaulttrainTransform method,
    which, however, cannot be applied to Lstdataset objects as it 'mysteriously' doesn't inherit a transform method from dataset.
    hence the problem to manually transform each batch before feeding it to the network'''
    train_transform = presets.yolo.YOLO3DefaultTrainTransform(width, height, net)
    train_dataloader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=False,  batchify_fn=batchify_fn,
                                                last_batch='discard')
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, last_batch='discard')




    # train
    for i, batch in enumerate(train_dataloader):
        with autograd.record():
            input_order = [0, 6, 1, 2, 3, 4, 5]
            obj_loss, center_loss, scale_loss, cls_loss = net(*[batch[o] for o in input_order])
            print(obj_loss, center_loss, scale_loss, cls_loss)





