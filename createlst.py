
import xml.etree.ElementTree as ET
import numpy as np
from imgresize import *


# TO DO ARG PARSER


class ReadFields:
    def __init__(self, imgname):
        self.w = self.h = self.d = float(0)
        self.xmn = self.ymn = self.xmx = self.ymx = float(0)
        self.imgname = imgname + '.xml'
        tree = ET.parse(self.imgname)
        root = tree.getroot()
        for size in root.findall('size'):
            for width in size.findall('width'):
                self.w = float(width.text)
            for height in size.findall('height'):
                self.h = float(height.text)
            for depth in size.findall('depth'):
                self.d = float(depth.text)
        for object in root.findall('object'):
            for bndbox in object.findall('bndbox'):
                for xmin in bndbox.findall('xmin'):
                    self.xmn = float(xmin.text)
                for ymin in bndbox.findall('ymin'):
                    self.ymn = float(ymin.text)
                for xmax in bndbox.findall('xmax'):
                    self.xmx = float(xmax.text)
                for ymax in bndbox.findall('ymax'):
                    self.ymx = float(ymax.text)

    def getfields(self):
        shape = [self.w, self.h, self.d]
        bndbx = [self.xmn, self.ymn, self.xmx, self.ymx]
        return shape, bndbx


def write_line(img_path, im_shape, boxes, ids, idx):
    h, w, c = im_shape
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
    str_path = [img_path + '.jpg']
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line


def create_lst(lst_file='val.lst', jpgpath='JPGImages/', xmlpath='Annotations/', resize=False, width=416, height=416):
    with open(lst_file, 'w') as fw:
        #n is the number of images in the dataset
        n = 141
        for i in range(0, 141):
            if i != 22 and i != 113 and i != 123 and i != 124:
                im_name = 'Alfa_Romeo_' + str(i)
                xml_image_data = xmlpath + im_name
                shape, bndBox = ReadFields(xml_image_data).getfields()
                id = np.array([float(0)])
                current_image = jpgpath + im_name
                if resize:
                    # image resizing using aux function
                    new_im_name = 'edit' + im_name
                    image_resize(current_image + '.jpg', new_im_name, width, height)
                    new_im_name = 'editImages/' + new_im_name
                    #print(bndBox)
                    bndBox[0] = (bndBox[0]*width)/shape[0]
                    bndBox[1] = (bndBox[1]*+height)/shape[1]
                    bndBox[2] = (bndBox[2]*width)/shape[0]
                    bndBox[3] = (bndBox[3]*height)/shape[1]
                    #print(bndBox)
                    line = write_line(new_im_name, np.array([width, height, 3]), np.array(bndBox), id, i)
                else:
                    line = write_line(current_image, np.array(shape), np.array(bndBox), id, i)
                    #print(line)
                fw.write(line)

if __name__ == '__main__':
    create_lst()
