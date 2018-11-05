"""Detection Dataset from LST file."""
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
from mxnet import gluon
import gluoncv as gcv


def _transform_label(label, height=None, width=None):
    label = np.array(label).ravel()
    header_len = int(label[0])  # label header
    label_width = int(label[1])  # the label width for each object, >= 5
    if label_width < 5:
        raise ValueError(
            "Label info for each object shoudl >= 5, given {}".format(label_width))
    min_len = header_len + 5
    if len(label) < min_len:
        raise ValueError(
            "Expected label length >= {}, got {}".format(min_len, len(label)))
    if (len(label) - header_len) % label_width:
        raise ValueError(
            "Broken label of size {}, cannot reshape into (N, {}) "
            "if header length {} is excluded".format(len(label), label_width, header_len))
    gcv_label = label[header_len:].reshape(-1, label_width)
    # swap columns, gluon-cv requires [xmin-ymin-xmax-ymax-id-extra0-extra1-xxx]
    ids = gcv_label[:, 0].copy()
    gcv_label[:, :4] = gcv_label[:, 1:5]
    gcv_label[:, 4] = ids
    # restore to absolute coordinates
    if height is not None:
        gcv_label[:, (0, 2)] *= width
    if width is not None:
        gcv_label[:, (1, 3)] *= height
    return gcv_label


class CustomDetection(Dataset):
    """Detection dataset loaded from LST file and raw images.
    LST file is a pure text file but with special label format.

    Checkout :ref:`lst_record_dataset` for tutorial of how to prepare this file.

    Parameters
    ----------
    filename : type
        Description of parameter `filename`.
    root : str
        Relative image root folder for filenames in LST file.
    flag : int, default is 1
        Use 1 for color images, and 0 for gray images.
    coord_normalized : boolean
        Indicate whether bounding box coordinates haved been normalized to (0, 1) in labels.
        If so, we will rescale back to absolute coordinates by multiplying width or height.

    """
    def __init__(self, filename, root='', flag=1, coord_normalized=True):
        self._flag = flag
        self._coord_normalized = coord_normalized
        self._items = []
        self._labels = []
        full_path = os.path.expanduser(filename)
        with open(full_path) as fin:
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                label = np.array(line[1:-1]).astype('float')
                im_path = os.path.join(root, line[-1])
                self._items.append(im_path)
                self._labels.append(label)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        im_path = self._items[idx]
        img = mx.image.imread(im_path, self._flag)
        h, w, _ = img.shape
        label = self._labels[idx]
        if self._coord_normalized:
            label = _transform_label(label, h, w)
        else:
            label = _transform_label(label)
        return img, label
