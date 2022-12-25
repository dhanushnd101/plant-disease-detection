import os
import numpy as np
import shutil
import random
rootdir = '/Users/dhanushdinesh/Documents/AML/PlantVillage_copy/'
classes = ['Pepper__bell___Bacterial_spot', 
           'Pepper__bell___healthy',
           'Potato___Early_blight',
           'Potato___healthy',
           'Potato___Late_blight',
           'Tomato__Target_Spot',
           'Tomato__Tomato_mosaic_virus',
           'Tomato__Tomato_YellowLeaf__Curl_Virus',
           'Tomato_Bacterial_spot',
           'Tomato_Early_blight',
           'Tomato_healthy',
           'Tomato_Late_blight',
           'Tomato_Leaf_Mold',
           'Tomato_Septoria_leaf_spot',
           'Tomato_Spider_mites_Two_spotted_spider_mite']


test_ratio = 0.25

for i in classes:

    os.makedirs(rootdir +'/train1/' + i)
    os.makedirs(rootdir +'/test1/' + i)
    
    source = rootdir + '/' + i
    
    allFileNames = os.listdir(source)
    np.random.shuffle(allFileNames)
    

    train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - test_ratio))])
    
    train_FileNames = [source+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]
    
    for name in train_FileNames:
      shutil.copy(name, rootdir +'/train/' + i)
    
    for name in test_FileNames:
      shutil.copy(name, rootdir +'/test/' + i)
    print("Done Copying "+ i)
  
  
print("Copying Done!")
