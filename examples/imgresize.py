
from __future__ import print_function
import sys
from PIL import Image
import mxnet as mx


def image_resize(img, out, width, height):
    im = Image.open(img)
    im.resize((width, height), Image.LANCZOS)
    save_path = 'editImages/' + out + '.jpg'
    im.save(save_path, "JPEG")


