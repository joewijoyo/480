import AlexNet
import sys
import tensorflow as tf
import GlobalVariables
import cv2
import numpy as np

def getImageData(imageFilePaths):
   imageDataList = []
   for imageFilePath in imageFilePaths:
      imageData = cv2.imread(imageFilePath)
      imageData = cv2.resize(imageData.astype(float), (227, 227)) #resize
      imageData = imageData.reshape((227, 227, 3))
      imageDataList.append(imageData)
   if len(imageDataList) < GlobalVariables.BATCH_LENGTH:
      padding = GlobalVariables.BATCH_LENGTH - len(imageDataList)
      for _ in range(padding):
         imageDataList.append([[[0] * 3] * 227] * 227)
   return imageDataList
               
#python predict.py ./predictImages/4.jpg ./predictImages/5.jpg
imageFilePaths = sys.argv[1:]
batch_x = getImageData(imageFilePaths) 
x = tf.placeholder("float", [GlobalVariables.BATCH_LENGTH, 227, 227, 3])
model = AlexNet.AlexNet(x, GlobalVariables.dropoutPro, len(GlobalVariables.possibleGenres))
#y = tf.placeholder(tf.float32, [None, len(GlobalVariables.possibleGenres)])
pred = model.fc3
y_ = tf.nn.softmax(pred) 
indexOfPrediction = tf.argmax(y_, 1)
saver = tf.train.Saver()


with tf.Session() as sess:
   saver.restore(sess, GlobalVariables.SAVED_SESSION_FILENAME)
   results = sess.run(y_, feed_dict={x: batch_x})
   for i in range(len(imageFilePaths)):
      predictionIndex = np.argmax(results[i])
      print "Prediction for image " + str(i+1) + " is : " + GlobalVariables.possibleGenres[predictionIndex]
