from gluoncv.data import VOCDetection
from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
import mxnet as mx
from mxnet import gluon
from gluoncv.loss import SSDMultiBoxLoss
from mxnet import autograd


# typically we use 2007+2012 trainval splits for training data
train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
# and use 2007 test as validation data
val_dataset = VOCDetection(splits=[(2007, 'test')])


width, height = 512, 512  # suppose we use 512 as base training size
utils.random.seed(233)  # fix seed in this tutorial

batch_size = 2  # for tutorial, we use smaller batch-size
# you can make it larger(if your CPU has more cores) to accelerate data loading
num_workers = 0


net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False)



x = mx.nd.zeros(shape=(1, 3, 512, 512))
net.initialize()
cids, scores, bboxes = net(x)

with autograd.train_mode():
    cls_preds, box_preds, anchors = net(x)


train_transform = presets.ssd.SSDDefaultTrainTransform(width, height, anchors)
batchify_fn = Tuple(Stack(), Stack(), Stack())
train_loader = DataLoader(
    train_dataset.transform(train_transform),
    batch_size,
    shuffle=True,
    batchify_fn=batchify_fn,
    last_batch='rollover',
    num_workers=num_workers)


mbox_loss = SSDMultiBoxLoss()
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})

for ib, batch in enumerate(train_loader):
    if ib > 0:
        break
    print('data:', batch[0].shape)
    print('class targets:', batch[1].shape)
    print('box targets:', batch[2].shape)
    with autograd.record():
        cls_pred, box_pred, anchors = net(batch[0])
        sum_loss, cls_loss, box_loss = mbox_loss(
            cls_pred, box_pred, batch[1], batch[2])
        # some standard gluon training steps:
        # autograd.backward(sum_loss)
        # trainer.step(1)