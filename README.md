![Logo](https://storage.googleapis.com/model_garden_artifacts/TF_Model_Garden.png)

# Welcome to the Model Garden for TensorFlow

The TensorFlow Model Garden is a repository with a number of different implementations of state-of-the-art (SOTA) models and modeling solutions for TensorFlow users. We aim to demonstrate the best practices for modeling so that TensorFlow users can take full advantage of TensorFlow for their research and product development.

## Structure

| Folder | Description |
|-----------|-------------|
| [official](official) | • **A collection of example implementations for SOTA models using the latest TensorFlow 2's high-level APIs**<br />• Officially maintained, supported, and kept up to date with the latest TensorFlow 2 APIs<br />• Reasonably optimized for fast performance while still being easy to read |
| [research](research) | • A collection of research model implementations in TensorFlow 1 or 2 by researchers<br />• Up to the individual researchers to maintain the model implementations and/or provide support on issues and pull requests |

## Contribution guidelines

If you want to contribute to models, please review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)

# Aim

In this project, it is aimed to classify and distinguish the objects in the image,
to determine the desired object and to determine how many of those objects are in the image.

# Method

In this project, training and testing were carried out on 4 classes. These classes are car, airplane, cat and dog. 
First of all, the model that made the training and test data available to the network was selected, 
then the RPN stage was arranged and the label map was created and given to the RPN network.
The training was carried out, tested and the accuracy of the network was calculated.
Implementation was done by following the steps below.

# Image Processing

In order for the images to be used in object detection, their size should be regulated and made black and white in accordance with the algorithm created. Data images were taken one by one, resized to 375x375, and rendered in grayscale with the help of the PIL library. Then, their names are edited and saved in a new specified folder.

![Screen Shot 2021-11-08 at 3 31 05 PM](https://user-images.githubusercontent.com/61979226/140821345-8400ebf0-72d3-4197-902c-ccb1eec144e8.png)

# Preparation of the data set

In order to train and test the network, the objects to be classified in the images must be selected, that is, their coordinates and classes must be given to the network ready.

There is an editor called LabelImg for this process.
Objects identified for each image in the dataset were manually marked one by one.

After the marking process, an xml file containing the types of objects in the picture and the coordinate information of the boxes containing the objects is obtained. This process is in the training data.
applied to all images found (6395 images). In order to determine the accuracy of the system, the same process was applied to all images (1549 images) in the test data as well as the training data.
In total, 6395+1549=7944 will be applied to the image.

![Screen Shot 2021-11-08 at 3 37 58 PM](https://user-images.githubusercontent.com/61979226/140822271-5c0e6897-8e40-4ac9-91f2-43e68d975d03.png)

# Folder Structure

To use the Tensorflow Api, the model-master structure was first downloaded from https://github.com/tensorflow/models.
After opening the model folder, a folder called scripts was created in the directory that appeared.
In this folder, the images in the data set are created with the LabelImg program.
Converted .xml files to .csv format and then to tfrecord format, moved to the research/object_detection folder and saved for use in education.

![Screen Shot 2021-11-08 at 3 40 51 PM](https://user-images.githubusercontent.com/61979226/140822576-78af4d7c-34c4-4770-9a81-f3efa53a27e8.png)

The content of the scripts directory has been arranged as seen in the image above. Here are the .csv files created in the data folder. 
There are two folders in the images folder, training and testing.
In these folders, there are related images and .xml files of these images. 80% of the data set is in the training folder and 20% is in the test folder. models\scripts\images\train : 80% of the dataset: 6395 images and 6395 .xml files will be used to train the model.
models\scripts\images\test : 20% of the dataset: 1549 images and 1549 .xml files will be used to test the model.
A total of 6395+1549=7944 images were used.

# Convert XML Files to CSV File

The xml_to_csv.py file was used for this process.
Some changes were made in the xml_to_csv.py file due to the file paths.

![Screen Shot 2021-11-09 at 11 32 53 PM](https://user-images.githubusercontent.com/61979226/141055880-49e34f7a-1874-45f7-903c-c34290fa1708.png)

The conversion to .csv format was performed for both the train and test file.
Train_labels.csv and test_labels.csv files were created in the data folder.
Then the .csv files need to be converted into tfrecord files. Because tensorflow takes files in tfrecord format as input to the model.

# Converting CSV Files to Record File

The generation_tfrecord.py file was used for this.
Some changes have been made for the model created in this file.
An integer value is sent for each object.
python3 generate_tfrecord.py --csv_input=data/train_labels.csv
--output_path=data/train.record --image_dir=images/
With the command, the train.record file was created in the data folder.
python3 generate_tfrecord.py --csv_input=data/test_labels.csv
--output_path=data/test.record --image_dir=images/
The test.record file was created in the data folder with the command.
Data and images folders after obtaining tfrecord files
Moved to models/research/object_detection. Thus, test, train images, xml, csv and record files were imported into the object_detection directory.

![Screen Shot 2021-11-09 at 11 39 47 PM](https://user-images.githubusercontent.com/61979226/141056377-1ad60bbd-c86b-4feb-a04e-96fa473d0237.png)

# Label Map – Creating a Label Map

TensorFlow requests to map each of the tags of the used objects to an integer value.
The object-detection.pbtxt file was manually created for this process.
Label_map.pbtxt file content:

![Screen Shot 2021-11-09 at 11 43 14 PM](https://user-images.githubusercontent.com/61979226/141056800-087a6133-7218-4706-b42d-b3ce1780f8f6.png)

Tag mapping has the extension .pbtxt. The tag map is used in the training phase.
After this file was created, it was moved to the training file.

# Configuration for RPN

There are specific structures for the RPN network. Faster_rcnn_resnet101_coco_2018_01_28, which is suitable for the model function used in the project, was chosen.
Then the config file is configured.
4 image classes are given as num_classes.
The image dimensions are given again as 375x375.

![Screen Shot 2021-11-09 at 11 57 02 PM](https://user-images.githubusercontent.com/61979226/141058086-01ec93d7-bfd3-4bb3-83a6-9cf5b086be6b.png)

Configured for training. (Learning step) The learning_rate parameters were determined and the num_steps value was set to 2000,
and the training was provided to be 2000 epochs.
The path to the model file is given.

![Screen Shot 2021-11-09 at 11 58 26 PM](https://user-images.githubusercontent.com/61979226/141058239-b6aa7ffd-3226-4d6d-b95b-c4d5b7a367fa.png)

Score_converter: set to SOFTMAX.
![Screen Shot 2021-11-09 at 11 59 13 PM](https://user-images.githubusercontent.com/61979226/141058319-ebc1697c-a4db-49d9-b5ca-ff94a7f557e1.png)

Logic score refers to each element for activation.
The config file should reach the object-detection.pbtxt and the dataset Record files.
object-detection.pbtxt: Feature map
record : The xml files of the images were first converted to csv format and then to a file in record format.
Eval_config: Used when running eval.py for testing.
Num_example: The number of test data available in the dataset is given.

![Screen Shot 2021-11-10 at 12 00 39 AM](https://user-images.githubusercontent.com/61979226/141058474-25676bd5-9abb-438d-84be-124e031fafdd.png)

# Train

Due to the Api structure used in the project, the training is carried out in the models/research/object_detection directory. Since many files are used during the training phase, when it comes to the research directory in the terminal, 
in order:
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
sudo python3 setup.py install

Protocol buffers are used to serialize structured data. Data becomes smaller and faster to use.
PYTHONPATH provides module search in defined directories.
Setup.py has functions such as deploying Python modules.
Due to the Tensorflow api structure used, these three commands are run in the given order. Then enter the object_detection directory and run the code below.

python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training
/faster_rcnn_resnet101_coco.config

Configured in the training file in this command
The faster_rcnn_resnet101_coco.config configuration file contains the object-detection.pbtxt property map.
Training is carried out by command.
The files created when the training takes place are also added to the training folder.
Graphical drawings were made using Tensorboard.

When the total loss is examined, it is seen that the model exceeds many local minimums and reaches the general minimum.

![Screen Shot 2021-11-10 at 12 06 03 AM](https://user-images.githubusercontent.com/61979226/141059080-c3e134a5-2642-4a07-a4a7-0e76284be5bb.png)

And in all losses graphs, it is seen that the loss amounts decrease.
Loss/BoxClassifierLoss/classification_loss/mul_1: It is the layer where the class of the object is determined.
Loss/BoxClassifierLoss/localization_loss/mul_1: Loss of the bounding box regressor ie the bounding coordinates of the object.
Loss/RPNLoss/localization_loss/mul_1: It is the loss of the bounding box regressor for the RPN, that is, the localization loss for the anchor.
Loss/RPNLoss/objectness_loss/mul_1:Classifier loss that classifies whether a bounding box is an object of interest or background.

# Frozen Model

export_inference_graph: It is a tool to export the object detection model for inference.
That is, it prepares the tensorflow object detection graph for inferences using the trained model. It allows us to use the trained model in testing or real life projects.
Run the following code to extract the frozen model from the trained model:

python export_inference_graph.py --input_type=image_tensor --
pipeline_config_path=training/pipeline.config --
trained_checkpoint_prefix=training/model.ckpt-2000 --
output_directory=export_graph

There is a trained model in the training folder mentioned here.
Pipeline.config from the trained model and the final model file model.ckpt-2000, which is the result of the training, is given.
The function of checkpoint is to hold checkpoints for the model. When this line is run, the export_graph directory is created and the frozen model is loaded into it.

#Test (Eval)

For the test phase, the frozen model is given as a checkpoint in the code line below and the resulting file is added to the eval directory.
python eval.py \ --logtostderr\ --pipeline_config_path=training /faster_rcnn_resnet101
_coco.config\ --checkpoint_dir= export_graph\ --eval_model_dir=eval/

![Screen Shot 2021-11-10 at 12 16 05 AM](https://user-images.githubusercontent.com/61979226/141060104-4bc420db-3b55-4e0e-bff3-d91acdbaab4c.png)

# Sample Outputs

Some sample outputs are obtained as follows.

![Screen Shot 2021-11-10 at 12 16 59 AM](https://user-images.githubusercontent.com/61979226/141060199-2105f114-abcb-4994-8d3f-ded26db24ea8.png)

# Accuracy Calculation
## RGB Setting
For counting the objects in the image, only 1549 images were selected from the image and xml files in the test directory.
The test_image folder has been created. Since the frozen model was saved as if it would work with RGB, 
the test data was manipulated and brought to RGB format.

## Accuracy Calculation
The images in the test_image folder were accessed by making changes in the algorithm created to access the images in the object_detection_runnig file.
For this process, first of all, the frozen model, test images, label in the object_detection_runnig file
Edited map paths. Then, to access all the images in the test_image folder in order, the image names were arranged as IMG_1.PNG, IMG2_.PNG IMG_1549.PNG.

With the for loop, the images are accessed one by one and the visualize_boxes_and_ labels_on_images _array function is called. 
This function is located in object_detection/utils/visualization_utils.py.
This file is actually used when running eval.py.

After the eval phase is over, we copied the visualization_utils.py file and made changes to the visualize_boxes_and_labels_on_images_array function in the new file. As a result of the changes made, it counts how many objects are found in the image and returns it as an array.
In order to perform this operation, firstly, a counter is started in the object_detection_running file, and the counter is incremented each time an image appears in the for, and it is given as a parameter to the visualize_boxes_and_labels_on_images_array function.

![Screen Shot 2021-11-10 at 12 21 28 AM](https://user-images.githubusercontent.com/61979226/141060802-7590f909-fe6c-4b1e-b999-ae65e02b1ff5.png)

In visualization_utils.py, an array with 1549 elements consisting of zeros and 4 0s in each element is created. In this array, the 4 zeros in each element represent classes.

![Screen Shot 2021-11-10 at 12 22 05 AM](https://user-images.githubusercontent.com/61979226/141060994-74daac85-6b9c-498c-86e4-13728b1f9324.png)

visualize_boxes_and_labels_on_images_array . The class_name inside this function is a variable that holds the class to which the object belongs. As seen in the picture below, by adding a for loop, the number of the element of the array is increased by one, whichever class the detected object belongs to. Thus, if there are two cars and a cat in the related image, the relevant element of the array will be [0 2 1 0], so the array predict returns.

![Screen Shot 2021-11-10 at 12 23 58 AM](https://user-images.githubusercontent.com/61979226/141061336-056ff1c9-6e79-4563-ba9d-cd4d7a03839a.png)

In the Object_detection_runnig.py file, in the same way, with a matrix of 1549x4 size, what type and how many objects are actually stored in the images used for the test. In order to keep the exact results in the csv file, an array containing 1549 lines and each element consisting of 4 0's was created. This array is used for the actual (accuracy) calculation. A 1549x9 matrix was created to list the actual and estimated values as well. In order to observe the correct values and predictions, an array containing 1549 elements and each element consisting of 9 0's was created. The first index of each element in this array will hold the image name followed by all classes as result and prediction.
![Screen Shot 2021-11-10 at 12 27 33 AM](https://user-images.githubusercontent.com/61979226/141061497-e0d659ae-135e-43a0-9292-ab277a430e42.png)

![Screen Shot 2021-11-10 at 12 27 41 AM](https://user-images.githubusercontent.com/61979226/141061501-2064d29b-f1d4-4b1d-930c-88b762bd6427.png)

The necessary addition for the table is performed to the tblt array and the variables are defined to calculate the results.
Using this piece of code, image-based and object-based accuracy rates were calculated for 4 different objects. In image-based accuracy, all objects in any image must be correctly detected. In the object-based calculation, accuracy rates are calculated by counting the number of objects and comparing them with the exact number of objects.
With the help of the for loop, the results produced using the frozen model created after the training with the array array are kept.
Each element of the array and the indices in each element are checked separately, and if the prediction produced is correct, the value of the relevant result variable is increased.

![Screen Shot 2021-11-10 at 12 29 11 AM](https://user-images.githubusercontent.com/61979226/141061637-39f127c9-6e41-47ed-934c-43612ba9af1c.png)

Total accuracy was first calculated from the results. Total accuracy kept in result_image variable count of individual objects of all classes in each image
includes the state of being correctly predicted. Therefore, dividing by 1549 gives us the total result.
In order to calculate how accurately each object is predicted individually, the result_... variables are divided by the exact value previously calculated for the relevant variable, and the accuracy for each class is calculated.

![Screen Shot 2021-11-10 at 12 41 17 AM](https://user-images.githubusercontent.com/61979226/141062976-1d9abfcf-8d42-46e1-a125-f619098c35e8.png)

Calculated accuracy values are saved both on the screen and in the result.txt file.
The created table is saved in the table.txt file and it can be examined in which picture the system produces and
what results should be.

![Screen Shot 2021-11-10 at 12 42 36 AM](https://user-images.githubusercontent.com/61979226/141063709-8bfe3c86-15dd-4bb9-931f-c8f84e0940bb.png)

# Results
##  Result.txt
![Screen Shot 2021-11-10 at 12 48 01 AM](https://user-images.githubusercontent.com/61979226/141063817-c4fbb332-4e22-4f50-a0b6-ac6553e8d829.png)

## Table.txt
![Screen Shot 2021-11-10 at 12 48 33 AM](https://user-images.githubusercontent.com/61979226/141063879-880f32ab-2701-4f62-8428-30716d27ecdd.png)

## Model Structure
![Screen Shot 2021-11-10 at 12 52 21 AM](https://user-images.githubusercontent.com/61979226/141064451-c31a2f0a-a650-4204-809b-428b3ae80c3b.png)



