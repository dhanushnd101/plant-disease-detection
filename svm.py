#   import os
import numpy as np
#import cv2 as cv
import pickle
from sklearn import svm
from sklearn import metrics

with open('./featDict.pickle', 'rb') as handle:
    featDict = pickle.load(handle)


trainX = []
trainY = []
testX = []
testY = []
count = []

#minCount =952
minCount = 800


trainRation=.8
print('Min length of the dataset:'+ str(minCount))


for key in featDict:
    print(key)

    if key!='Potato___healthy' and key!= 'Tomato__Tomato_mosaic_virus' and key!= 'PlantVillage':
        ImgNames = []
        for count, img_name in enumerate(featDict[key]):
            ImgNames.append(img_name)
        np.random.shuffle(ImgNames)
        i = 0
        for names in ImgNames:
            i+= 1
            if(i < minCount*trainRation):
                trainX.append(featDict[key][names])
                trainY.append(key)
            else:
                testX.append(featDict[key][names])
                testY.append(key)

clf = svm.SVC(kernel='linear')

print("completed linear SVM")
#clf = svm.SVC(decision_function_shape='ovo')

print(len(testX),len(testY),len(trainX),len(trainY))

'''
clf.fit(trainX, trainY)

print("Completed SVM fit")
#print(clf.predict([testX[1]]))
#print([testY[1]])

y_pred = clf.predict(testX)
print("Accuracy:",metrics.accuracy_score(testY, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
#precision_s = metrics.average_precision_score(testY, y_pred)
#print("Precision: " + str(precision_s))

#recall_s = metrics.recall_score(testY, y_pred, average="weighted")
#print("Recall: " + str(recall_s))

#f1score = (2*precision_s*recall_s)/(precision_s+recall_s)
#print("F1 score: "+ str(f1score))

matrix_confusion = metrics.confusion_matrix(testY, y_pred)

print(matrix_confusion)

import matplotlib.pyplot as plt

metrics.plot_confusion_matrix(clf, testX, testY)  
plt.show()
'''