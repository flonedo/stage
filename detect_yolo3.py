#!/bin/env python
from gluoncv import data
from mxnet import gluon
from PIL import Image, ImageDraw, ImageFont
import load_classes
import os
import argparse
from gluoncv.model_zoo import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Detect objects with YOLO networks.')
    parser.add_argument('--network', type=str, default="../yolo-symbol.json",
                        help='Path to network architecture json file.')
    parser.add_argument('--params', type=str, default="../yolo-0015.params",
                        help='Path to network parameters params file.')
    parser.add_argument('--synset', type=str, default='../yolo3_darknet53_custom/synset.txt',
                        help='Path to class names txt file.')
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape, use 320, 416, 608...")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold score for class detection.')
    parser.add_argument('--input-image', type=str, default="../test/esempio.jpg",
                        help='Path to image to detect.')
    parser.add_argument('--output-folder', type=str, default='../detected_images',
                        help='Path to output folder.')
    parser.add_argument('--output-file', type=str, default='out',
                        help='Output file name.')
    args = parser.parse_args()
    return args


def __get_bounding_box_scale(resize, width, height):
    shortest = min(width, height)
    return 1.0/(float(resize)/float(shortest))

def detect(network_path, params_path, synset, image_path, output_folder, output_image_name, size=416, threshold=0.5):
    # loads classes from the synset network export file
    classes = load_classes.get_dataset_classes(synset)
    #loads the network from .json and .params network export files
    net = gluon.nn.SymbolBlock.imports(network_path, ['data'], params_path)

    # applies transformation to image
    x, img = data.transforms.presets.yolo.load_test(image_path, short=size)
    # detects
    class_IDs, scores, bounding_boxes = net(x)
    # creates an output folder if it doesn't already exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # loads the transformed and resized image to draw bounding boxes and class labels
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    original_width, original_height = image.size
    scale = __get_bounding_box_scale(size, original_width, original_height)
    # it shows the best 5 results
    for i in range(0, 4):
        xmin = scale*((bounding_boxes[0][i][0]).asscalar())
        ymin = scale*((bounding_boxes[0][i][1]).asscalar())
        xmax = scale*((bounding_boxes[0][i][2]).asscalar())
        ymax = scale*((bounding_boxes[0][i][3]).asscalar())
        score = float((scores[0][i]).asscalar())
        id = int((class_IDs[0][i]).asscalar())
        cl = classes[int(id)]
        label = 'class: ' + cl + ' score: ' + str(score)
        print(label)
        print('xmin ', xmin, 'ymin ', ymin, 'xmax ', xmax, 'ymax ', ymax)
        # score threshold for detected objects to be drawn
        if score > threshold:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=128)
            draw.text((xmin, ymin), label, font=font, fill=128)
    del draw
    image.save(os.path.join(output_folder,  output_image_name + ".png"), "PNG")


def yolo3_detect():
    args = parse_args()
    detect(args.network, args.params, args.synset, args.input_image, args.output_folder, args.output_file, args.data_shape,
           args.threshold)


if __name__ == '__main__':
    yolo3_detect()
