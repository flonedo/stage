# Model training and object detection with YOLO v3

## Installation
 To run YOLO v3 training and detection Python scripts youl'll need Mxnet ([Note 1](.#Note-1:)) framework and GluonCV ([Note 2](.#Note-2:)) toolkit.
 Mxnet works with both Python 2 and Python 3, but we suggest using Python 2.7 as Python 3 can sometimes be troublesome.
 We suggest installing the following versions of Mxnet framework and GluonCV toolkit:

```sh
$ pip install gluoncv==0.4.0b20181023
$ pip install mxnet==1.3.0
```
If you want to train your model on GPU, please install Mxnet for GPU. In the example below we install Mxnet with Cuda 9 support:
```sh
$ pip install gluoncv==0.4.0b20181023
$ pip install mxnet-cu90==1.3.0
```
You can check on [PyPi](https://pypi.org/project/mxnet/) for the Mxnet version you need.

###### __Note 1:__
GluonCV requires Mxnet version 1.3.0 and higher.
###### __Note 2:__
GluonCV 0.3.0 stable release contains a known bug that prevents our training script training from working; please use a later version of the toolkit (version 0.4.0b20181023 or later).

## Model training
### Dataset creation
To train your YOLO v3 model you need a dataset in LST file format ([Note 3](-#Note-3:)); such file can be easily created by running functions provided by the *dataset_creator.py* module.
To create your datasets from a database export json lines file:

1. To create the LST file run the __create_dataset_from_json(json_file_path, output_path)__ function in the module.
 INPUT:
   * __json_file_path__: path to your database export json lines file. The requiered structure for this file is described below ([Note 4](.#Note-4:)).
   * __output_path__: path to the folder where you want to save your datasets.

   OUTPUT:
   * For each client id found in the input JSON LINES file, the function saves a *clientId.lst* file containing the entire dataset for that client id; such file is saved in a *clientId* folder in the provided *output_path*.
   * When the function terminates successfully *"end"* will be displayed on console.
2. To split a LST dataset into training dataset and validation dataset, run the __split_dataset(dataset_path, lst_file_name, synset_path, ratio)__ function in the module.
 INPUT:
   * __dataset_path__: path to the folder containing the LST dataset file you want to split into training and validation.
   * __lst_file_name__: name of the LST file you want to split.
   * __synset_path__: path to the synset TXT file corresponding to your dataset classes ([Note 5](.#Note-5:)).
   * __ratio__: ratio in the (0, 1) range to split your dataset into training and validation. We suggest using a ratio of 0.7 (70% training and 30% validation).

   OUTPUT:
   * Randomly splits your dataset into *train.lst* and *valid.lst* in the given ratio; both file are saved in the the same folder as the input *clientId.lst*, which remains unvaried.
   * Additionally it creates a *dataset_stats.txt" file containing statistics about the split dataset; such files states:
   * __Dataset cardinality__: the arithmetic means of the number of labels for each image. A lower cardinality is preferred.
   * __Dataset density__: the arithmetic means of the number of labels for each image divided by the number of classes. A higher density is preferred.
   * __Training/validation ratio__: the split ratio. A ratio of 70% training and 30% validation is preferred.
   * __total images number__
   * __training images number__
   * __validation images number__
###### __Note 3:__
A LST file can be read as a tab separated CSV file marked with the *.lst* extension.
###### __Note 4:__
Each line in the database export JSON LINES file represent an image in the dataset and should follow the following structure:
```json
{"id":"some_image_id", "clientId":"some_client_id", "url":"url_to_image", "data":{"channels":3, "width":100, "height":100, "classes":[{"class":{"className":0.0}, "boundingBox": [0.0, 0.0, 0.0, 0.0]}]}}
```
where:
- __id__: is the image id in string format.
- __clientId__: is the client id in string format.
- __url__: is the Url where the image is hosted in string format.
- __data__: information required for the training.
 * __channels__: number of channels in which the image is encoded where 3 stands for RGB image and 1 stands for BW image; as of now our YOLO v3 training script only support color images (3 channels).
 * __width__: the image width in integer format.
 * __height__: the image height in integer format.
 * __classes__: an array containing the bounding boxes delimiting the objects of interest in the image; each element in the array is structured as follow:
 * __class__: a json as follows:
 * __className__: class name in float format.
 * __boundingBox__: array containing the coordinates of the bounding box delimiting the object of interest in the order *xmin*, *ymin*, *xmax*, *ymax*. Each coordinate is expressed in float format.

###### __Note 5:__
A synset file is a TXT file that maps string class names with their number encoding. Here is a sample *synset.txt*:
```txt
cat
dog
rabbit
guinea pig
```
where cat corresponds to className = 0.0, dog to className = 1.0, rabbit to className = 2.0 and guinea pig to className = 3.0. The synset TXT file must be created manually.

### Training
Once you created your custom dataset, you can train a YOLO v3 model on it. We suggest using a GPU since training is extremely time expensive (e.g. the training process on a toy dataset of 10 classes with 40 samples each would run for over 48 hours on CPU).
To train your model we provide the *train_yolo3.py* script. You can fine tune the training hyperparamers by providing your required values as arguments while launching the script; supported arguments are:
- __network__: base network name which serves as a feature extraction base. Defaults to *darknet53*.
- __data-shape__: input data shape; the shortest side of each image will be resized to match this value. Accepted values are 320, 416 and 608 (32 multiples) and it defaults to 416.
- __batch_size__: number of images in each computing batch. It must be tuned accordingly to your hardware as a batch size to big would trigger a bus error. It defaults to 8 (which works for CPU), but with GPU training you might be able to use a bigger size such as 32.
- __dataset__: the dataset on which you train your model; please use "custom" for training on a custom dataset. It defaults to "custom".
- __train-dataset__: path to your training dataset LST file.
- __valid-dataset__: path to your validation dataset LST file.
- __synset__: path to your synset TXT file.
- __num_workers__: number of workers used for training. You must tune it accordingly to your hardware as a number too high mught result in a thread error. It defaults to 0.
- __gpus__: numbers of the gpus to use as a comma separated list of integers; to train on CPU leave this argument empty. Defaults to empty list.
- __epochs__: number of training epochs. Defaults to 200.
- __resume__: path to a PARAMS saved checkpoint in case you want to resume training. Defaults to empty string (no resume).
- __start-epoch__: epoch from which you want to resume training in case you are loading a PARAMS checkpoint with the *resume* argument.
- __lr__: learning rate. Speed at which your model learns the dataset content. A high learning rate makes you model learn quicker, but might result in overfitting. Defaults to 0.001.
- __lr-decay-rate__: decay rate of learning rate. Defaults to 0.1.
- __lr-decay-epoch__: epochs at which your learning rate decays. Defaults to '160,180'.
- __momentum__: stochastic gradient descent momentum. Defaults to 0.9.
- __log-interval__: logging mini batch training stats interval. Defaults to 100.
- __save-prefix__: saving parameters prefix. Defaults to empty string.
- __save-interval__: saving parameters epoch interval. Defaults to 10.
- __val-interval__: validation epoch interval. Increasing the number will reduce training time if validation is slow. Defaults to 1.
- __seed__: seed to fix the behaviour of random function. Defaults to 233.
- __num samples__: number of images in your training dataset. If not provided defaults to -1 and the number will be automatically inferred.
- __export-epoch__: epoch in which you want to export your network model in json format and its parameters.
- __syncbn__: argument to synchronize BN across devices for distributed training.

#### Fine tuning a model
By default our *train_yolo3.py* script fine tunes a YOLO model pre-trained on VOC dataset. Fine tuning a pre-trained model reduces the training time, meaning that compared to training from scratch you'll be able to acquire the same accuracy in less epochs. The best way to fine tune a model is to load only the pre-trained network weights and skip loading the additional network layers: you can do so by setting *pretrained_base = False* and *transfer = 'some-weights'* when loading the network with the *get_model(name,* ***kwargs)* function; as stated above, by default our YOLO training script fine tunes on VOC, but you can change it to load COCO weights instead.
To fine tune a model run the script as follows:
```sh
$python train_yolo3.py --train-dataset path/to/train.lst --valid-dataset path/to/valid.lst --synset path/to/synset.txt
```
#### Resume training
You can resume training by loading a PARAMS file ([Note 6](.#Note-6:)); to do so you can launch the script as follows:
```sh
$python train_yolo3.py --train-dataset path/to/train.lst --valid-dataset path/to/valid.lst --synset path/to/synset.txt --resume pat/to/parameters-0150.params --start-epoch 150
```
###### __Note 6:__
PARAMS files saved by the export function cannot be loaded this way as they are meant to work only with the JSON file describing the network they come with; this happens because when exporting the trained model, additional layers are attached to the network so in the end it differs from the "vanilla" network provided by the *get_model* function we use in the script.
To load the weights with the *load_parameters* function we use in the script, you should use one of the PARAMS files saved every *save-interval* by the *save_params* function.

#### Reading the validation results
To evaluate the accuracy of our model during the validation process we use the *mean Average Precision* (mAP) metric, set with an *Intersection Over Union* (iou) threshold of 0.5. The goal of this metric is to calculate how well the detected bounding box overlaps with the ground truth for each class.
The output of the validation process is a mAP score for each tested class; a mAP score is float in the range (0,1) where a mAP value closer to 1 means a better accuracy.

#### Exporting the trained model
As mentioned above an *export* function in the training script exports the trained network to file during the set *export-epoch*. The export function creates two files to save the trained network:
- __network-symbol.json__: a JSON file describing the network layers. By default the file is saved as network-symbol.json.
- __network-0000.params__: a binary PARAMS file containing the network weights. Buy default the file is saved as network-export_epoch.params where *export_epoch* is an integer representing the epoch in which the network is exported.

A model exported this way can be loaded for later use by both python applications (e.g. our detection script) and application written in other languages supported by the Mxnet framework. Note that the exported network-symbol.json and network-0000.params are strictly coupled and can only be used together.

## Detection
Once you trained and exported your network, you can use it for detection. As explained above, an exported network model can be loaded by applications written in both Python and other Mxnet supported languages. The *detect_yolo3.py* script we provide is written in Python.
Contrarily to training, unless high performance is needed (e.g. real time video detection), detection scripts can be reasonably run on CPU. For faster results on CPU we suggest trying MKLDNN (Math Kernel Library for Deep Neural Networks) supporting hardware and Mxnet version.
Our detection script takes the following arguments:
- __network__: path to the JSON file that describes the network structure to load.
- __params__: path to the PARAMS file that contains the trained model weights to load.
- __synset__: path to the TXT file containing the synset classes to load.
- __data_shape__: input data shape; the shortest side of input image will be resized to this before detection. Accepted values are 320, 416, 608 (32 multiples) and it defaults to 416. Detection data shape can be different from training data shape.
- __threshold__: confidence score to consider a detected object a true positive. Bounding boxes for objects detected with a score lower that threshold will be discarded.
- __input_image__: path to the image to run detection upon.
- __output_folder__: path to the folder where to save output image.
- __output_image__: name of the output image.

#### Run detection on an image
To detect objects you can run our *detect_yolo3.py* script as follows:
```sh
$python detect_yolo3.py --network network-symbol.json --params network-0000.params --synset path/to/synset.txt --input-image path/to/image.jpg --output-folder path/to/folder --output-file image_copy
```
Running this script provides two outputs:
- class, confidence score and bounding box for each detected object will be printed on console.
- For each detected object whose score is above threshold, the corresponding class name, confidence score and bounding box will be drawn on a copy of the original image; such image will be saved in the given *output_folder* as a PNG file with the *output_image* name.

## Utilities
### load_classes.py module
This module is used by both the training script and the detection script. The *def get_dataset_classes(synset_path)* reads the classes written into the synset TXT file and loads them into an array of strings.
