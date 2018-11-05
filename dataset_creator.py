import json
import os
import csv
import load_classes
import numpy as np


def write_line(img_path, im_shape, boxes, ids, idx):
    h, w = im_shape
    A = 4
    B = 5
    C = w
    D = h
    labels = np.hstack((ids, boxes)).astype(float)
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path + '.jpg']
    # returns a tab separated lst file line (readable as tab separated csv)
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line


def create_split_dataset_stats_file(lst_file_path, class_cardinality_data, labels_per_image, classes):
    classes_number = len(classes)
    dataset_images_number = len(labels_per_image)
    # writes dataset stats into a file
    with open(os.path.join(lst_file_path, 'dataset_stats.txt'), 'w') as stats:
        stats.write('Number of images: ' + str(dataset_images_number) + '\n')
        stats.write('Number of classes: ' + str(classes_number) + '\n')
        total_image_labels_number = 0
        total_image_labels_number_over_classes = 0
        # calculates cardinality and density of the dataset
        for key in labels_per_image:
            label_number = labels_per_image.get(key)
            total_image_labels_number += label_number
            total_image_labels_number_over_classes += label_number/classes_number
        # datased cardinality is the arithmetic means of the number of labels for each image
        dataset_cardinality = total_image_labels_number/dataset_images_number
        # dataset density is the arithmetic means of the number of labels for each image divided by the number of classes
        dataset_density = total_image_labels_number_over_classes/dataset_images_number
        stats.write('Dataset cardinality: ' + str(dataset_cardinality) + '\n')
        stats.write('Dataset density: ' + str(dataset_density) + '\n')
        stats.write('Training/Validation ratio: ' + str(ratio) + '\n')
        index = 0
        for element in class_cardinality_data:
            stats.write('class: ' + (classes[index]).rstrip() + ' total images number: ' + str(element[0]) +
                        ' training images number: ' + str(element[1]) + ' validation images number: ' +
                        str(element[2]) + '\n')
            index += 1


# splits a dataset into training dataset and validation dataset and creates a dataset stats txt file
def split_dataset(dataset_path, lst_file_name, synset_path, ratio):
    np.random.seed(1357)
    classes = load_classes.get_dataset_classes(synset_path)
    dim = len(classes)
    labels_per_image = dict()
    # table to save the number of images for each class in total, in the train dataset and in the valid dataset
    class_cardinality_data = np.zeros((dim, 3))
    with open(os.path.join(dataset_path, lst_file_name)) as dataset:
        with open(os.path.join(dataset_path, 'train.lst'), "w") as train_lst_file:
            with open(os.path.join(dataset_path, 'valid.lst'), "w") as valid_lst_file:
                dataset_rows = csv.reader(dataset, delimiter='	', quotechar='"')
                for row in dataset_rows:
                    image_id = row[0]
                    image_class = int(float(row[5]))
                    class_cardinality_data[image_class][0] += 1
                    if image_id in labels_per_image:
                        labels_per_image[image_id] += 1
                    else:
                        labels_per_image[image_id] = 1
                    r = np.random.rand(1)
                    # assigns images into train and valid datasets according to ratio
                    # and (class_cardinality_data[image_class][1] <= class_cardinality_data[image_class][0] * ratio or
                    # class_cardinality_data [image_class][2] > class_cardinality_data [image_class][0] * (1-ratio))
                    if r < ratio:
                        train_lst_file.write('\t'.join(row) + '\n')
                        class_cardinality_data[image_class][1] += 1
                    else:
                        valid_lst_file.write('\t'.join(row) + '\n')
                        class_cardinality_data[image_class][2] += 1
    create_split_dataset_stats_file(dataset_path, class_cardinality_data, labels_per_image, classes)
    print('end')


def parse_json(line, output_path):
    # reads fields from a single image entry in the json lines file
    line_data = line["data"]
    image_id = line["id"]
    client_id = line["clientId"]
    url = line["url"]
    # parameter channels is not used
    channels = line_data["channels"]
    width = line_data["width"]
    height = line_data["height"]
    # reads bounding boxes for each class present in a single image entry
    for item in line_data["classes"]:
        obj_class = item["class"]
        class_name = obj_class["className"]
        b_box = item["boundingBox"]
        lst_line = write_line(url, np.array([width, height]), np.array(b_box), np.array([float(class_name)]), image_id)
        # enters the output folder to create a nested structure
        os.chdir(output_path)
        # creates a new folder to save the lst file if it doesn't already exist
        if not os.path.exists(client_id):
            os.makedirs(client_id)
        # goes back to the main folder
        os.chdir('..')
        lst_file_name = client_id + '.lst'
        lst_file_path = os.path.join(output_path, client_id, lst_file_name)
        # writes a new line in the lst file and creates it if it doesn't already exist
        with open(lst_file_path, "a") as lst_file:
            lst_file.write(lst_line)


def create_dataset_from_json(json_file_path, output_path):
    # creates an output folder if it doesn't already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # opens the json lines file to to convert to lst format
    with open(json_file_path, 'r') as json_lines:
        # json lines is read one line (one json) at a time
        for line in json_lines:
            # labels_per_image counts how many labels are associated with each image
            parse_json(json.loads(line), output_path)
    print('end')

if __name__ == '__main__':
    input_path = "json_line_structure.json"
    output_path = r'output'
    synset_path = 'yolo3_darknet53_custom/synset.txt'
    create_dataset_from_json(input_path, output_path)
    ratio = 0.7
    split_dataset('output/clientId/', 'clientId.lst', synset_path, ratio)
    print('end')

