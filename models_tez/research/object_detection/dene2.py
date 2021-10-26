import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
import pickle

import glob
from tqdm import tqdm

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util


MODEL_NAME = 'exported_graph'
PATH_TO_CKPT=MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/object-detection.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'test_images'
NUM_CLASSES = 4


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,   use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3))

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'IMG_{}.jpg'.format(i)) for i in range(1, 1549) ]

IMAGE_SIZE = (375, 375)

counter=-1

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      counter=counter+1

      dene=vis_util.visualize_boxes_and_labels_on_image_array( 
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          say=counter,
          use_normalized_coordinates=True)

      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
    
        


col_list=["filename", "class"] 
tlb=pd.read_csv('data/test_labels.csv',usecols=col_list)  
len(tlb['filename'])

fileID=0
pre_fn=tlb['filename'][0]
array = [ [ 0 for i in range(4) ] for j in range(1549) ] 
print(len(array))
print(array)

array[0][2]=2
numAirplane=0
numCar=0
numCat=2
numDog=0

for ctr in range(2,len(tlb['filename'])):

  cur_fn = tlb['filename'][ctr]
  if  cur_fn not in pre_fn:
    fileID = fileID+1
  
  cur_class = tlb['class'][ctr]  
  if  cur_class in "airplane":
    array[fileID][0]=array[fileID][0]+1
    numAirplane=numAirplane+1
  elif cur_class in "car":
    array[fileID][1]=array[fileID][1]+1
    numCar=numCar+1
  elif cur_class in "cat":
    array[fileID][2]=array[fileID][2]+1
    numCat=numCat+1
  elif cur_class in "dog":
    array[fileID][3]=array[fileID][3]+1
    numDog=numDog+1
  pre_fn = cur_fn

sonuc_image=0
sonuc_airplane=0
sonuc_car=0
sonuc_cat=0
sonuc_dog=0

for x in range(0,1549):
    if(array[x]==dene[x]): 
        sonuc_image=sonuc_image+1
    if array[x][0]>=dene[x][0]:
       sonuc_airplane=sonuc_airplane+dene[x][0]
    else:
       sonuc_airplane=sonuc_airplane+array[x][0]
    if array[x][1]>=dene[x][1]:
        sonuc_car=sonuc_car+dene[x][1]
    else:
        sonuc_car=sonuc_car+array[x][1]
    if array[x][2]>=dene[x][2]:
        sonuc_cat=sonuc_cat+dene[x][2]
    else:
        sonuc_cat=sonuc_cat+array[x][2]
    if array[x][3]>=dene[x][3]:
        sonuc_dog=sonuc_dog+dene[x][3]
    else:
        sonuc_dog=sonuc_dog+array[x][3]


sonuc_image=sonuc_image/1549
sonuc_airplane_=sonuc_airplane/numAirplane
sonuc_car_=sonuc_car/numCar
sonuc_cat_=sonuc_cat/numCat
sonuc_dog_=sonuc_dog/numDog

sonuc_genel=(sonuc_airplane+sonuc_car+sonuc_cat+sonuc_dog)/(numAirplane+numCar+numCat+numDog)

print("Total sonuc", sonuc_image)
print("Total airplane", sonuc_airplane_)
print("Total car", sonuc_car_)
print("Total cat", sonuc_cat_)
print("Total dog", sonuc_dog_)
print("Total genel", sonuc_genel)

with open('var_array_dene_acc.pkl', 'w') as f:
    pickle.dump([array, dene], f)


        
        








''' for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      taco = [category_index.get(value) for index,value in numerate(classes[0]) if scores[0,index] > 0.8]
      #print(len(taco))
      counter=counter+1
     
      bu kisim calisan
      obj_len=len(taco)
     
     obj=taco
      f = open("deneme2.txt", "a",encoding="utf-8")
      f.write(str(obj))
      f.write(". objects \n")    
      f.close()
    print("object", len(taco))
     print(taco)
     f.close()
  '''