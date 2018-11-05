from __future__ import print_function
import mxnet as mx
from mxnet import gluon
import argparse
import os
import tarfile
import gluoncv as gcv
from gluoncv.data import LstDetection
from gluoncv.data.transforms import presets
from gluoncv import utils
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
from mxnet import autograd
from mxnet.gluon.data import Dataset
from multiprocessing import cpu_count
import numpy as np
from PIL import Image




image_file = Image.open("Alfa_Romeo_94.jpg") # open colour image
print(image_file.format, '/n',  image_file.size, '/n', image_file.mode, '/n', image_file.getchannel)
image_file = image_file.convert('1') # convert image to black and white

image_file = image_file.convert(mode='L')

image_file.save('bwimage.png')



'''def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


img_data = load_image('')'''



# checks if dataset is correctly initialized
'''def dataset_test(train_dataset):
    print('length:', len(train_dataset))
    #second_img = train_dataset[1][0]
    #print('image shape:', second_img.shape)
    #print('Label example:')
    #print(train_dataset[1][1])
    train_image, train_label = train_dataset[0]
    bboxes = train_label[:, :4]
    cids = train_label[:, 4:5]
    print('image:', train_image.shape)
    print('bboxes:', bboxes.shape, 'class ids:', cids.shape)'''


# arg parse
def parse_args():
    parser = argparse.ArgumentParser(description='Train yolo networks with custom datasets.')
    parser.add_argument('--dataset', type=str, default='val.lst',
                        help="Brands dataset.")
    parser.add_argument('--imgpath', type=str, default='JPGImages/',
                        help='Path to images folder')
    parser.add_argument('--batchsize', type=int, default=10,
                         help='training batch size')
    args = parser.parse_args()
    return args


# main
args = parse_args()
batch_size = 32 #args.batchsize
ctx = mx.cpu()

#train_dataset = mx.image.ImageIter(batch_size=4, data_shape=(3, 224, 224), label_width=1,
                                   #path_imglist='val.lst', aug_list=[mx.image.ForceResizeAug(224, 224)])
#valid_dataset = mx.image.ImageIter(batch_size=4, data_shape=(3, 224, 224), label_width=1,
                                   #path_imglist='val.lst', aug_list=[mx.image.ForceResizeAug(224,224)])


def write_line(img_path, im_shape, boxes, ids, idx):
    h, w = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids, boxes)).astype(float)
    #print(labels)
    # normalized bboxes (recommanded)
    # labels[:, (1, 3)] /= float(w)
    # labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path + '.png']
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line


with open('mock-dataset.lst', 'w') as ds:
    for i in range(0, 100):
        img = 'bwimage'
        id = np.array([float(0)])
        line = write_line(img, [28, 28], np.array([1, 1, 26, 27]), id, i) #line = write_line(currentImg, shape, np.array(bndBox), id, i)
        #print(line)
        ds.write(line)



train_dataset = LstDetection('mock-dataset.lst', root=os.path.expanduser('.'), flag=0, coord_normalized=False)
valid_dataset = LstDetection('mock-dataset.lst', root=os.path.expanduser('.'), flag=0, coord_normalized=False)








# PERCHE I TIPI IN PYTHON NON HANNO SENSO! LSTDETECTION E' SOTTOTIPO DI DATASET QUINDI I METODI DI DATASET DOVREBBERO FUNZIONARE --> LISKOV SUBSTITUTION PRINCIPLE DEL SOLID
#train_loader = DataLoader(Dataset.transform(train_dataset, train_transform, lazy=False), batch_size, shuffle=True,
#                          batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

#y_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
train_data_loader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=False, last_batch='discard')#, batchify_fn=y_batchify_fn)
valid_data_loader = mx.gluon.data.DataLoader(valid_dataset, batch_size, shuffle=True, last_batch='discard')#, batchify_fn=y_batchify_fn)

#dataset_test(train_dataset)
#print(train_dataset[0][0])
#print(train_dataset[0][1])




#NET AND TRAINING
def construct_net():
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(128, activation="relu"))
        net.add(gluon.nn.Dense(64, activation="relu"))
        net.add(gluon.nn.Dense(10))
    return net


ctx = mx.cpu()

net = construct_net()
net.hybridize()
net.initialize(mx.init.Xavier(), ctx=ctx)

criterion = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})


epochs = 5
for epoch in range(epochs):
    # training loop (with autograd and trainer steps, etc.)
    cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
    training_samples = 0
    for batch_idx, (data, label) in enumerate(train_data_loader):
        #print(data)
        data = data.astype('float32')/255
        data = data.as_in_context(ctx).reshape((-1, 784)) #48400)) # 220*220=784
        label = mx.nd.zeros((32))
        #print(label.size)
        #print(label)
        #print(label)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            #print('OUTPUT DATA:')
            #print(output.shape)
            #print(output)
            loss = criterion(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_train_loss += loss.sum()
        training_samples += data.shape[0]
    train_loss = cumulative_train_loss.asscalar()/training_samples

    # validation loop
    cumulative_valid_loss = mx.nd.zeros(1, ctx)
    valid_samples = 0
    for batch_idx, (data, label) in enumerate(valid_data_loader):
        #print(data)
        data = data.as_in_context(ctx).reshape((-1, 784)) #48400)) # 220*220=784
        data = data.astype('float32')/255
        #print(data)
        label = mx.nd.zeros((32))
        label = label.as_in_context(ctx)
        output = net(data)
        loss = criterion(output, label)
        cumulative_valid_loss += loss.sum()
        valid_samples += data.shape[0]
    valid_loss = cumulative_valid_loss.asscalar()/valid_samples

    print("Epoch {}, training loss: {:.2f}, validation loss: {:.2f}".format(epoch, train_loss, valid_loss))

for x_batch, y_batch in train_data_loader:
    print("X_batch has shape {}, and y_batch has shape {}".format(x_batch.shape, y_batch.shape))




#classes = ['0']
###net = model_zoo.yolo3_darknet53_custom(classes=classes, pretrained_base=False, pretrained=False)
#net.initialize(mx.init.Xavier(), ctx=ctx)



