import pandas as pd
import numpy as np

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
    array[fileID][3]=array[fileID][3]+1 #burda da dizideki yani mesela 10. resimdeki toplam kopek sayisi 
    numDog=numDog+1 #burda mesela dog varsa degeri 1 artiyor ya genel kopek sayisi
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
       sonuc_airplane=sonuc_airplane+dene[x][0] #
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

print("Total sonuc", sonuc_image) #Goruntu bazli dogruluk
print("Total airplane", sonuc_airplane_) #Nesne bazli ucak nesnesine gore dogruluk
print("Total car", sonuc_car_) #Nesne bazli car a gore
print("Total cat", sonuc_cat_) #Nesne bazli cat e gore
print("Total dog", sonuc_dog_) #Nesne bazli dog a gore
print("Total genel", sonuc_genel) #Nesne bazli genel hepsi

