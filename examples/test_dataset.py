from yolo3.customdetection import *
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
from gluoncv.data.transforms import presets

batch_size = 2  # for tutorial, we use smaller batch-size
num_workers = 0  # you can make it larger(if your CPU has more cores) to accelerate data loading

train_dataset_lst_file = 'output_1/clientId/clientId_train.lst'
train_dataset = CustomDetection(train_dataset_lst_file, root=os.path.expanduser('.'), flag=1, coord_normalized=False)

net = net = model_zoo.get_model('yolo3_darknet53_voc', pretrained_base=False)
net.initialize()
# behavior of batchify_fn: stack images, and pad labels
train_transform = presets.yolo.YOLO3DefaultTrainTransform(416, 416, net)
# return stacked images, center_targets, scale_targets, gradient weights, objectness_targets, class_targets
# additionally, return padded ground truth bboxes, so there are 7 components returned by dataloader
batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True,
                          batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
print(train_loader)