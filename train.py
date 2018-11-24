import tensorflow as tf
import numpy as np
import AlexNet
import os
import cv2
import utilities
import GlobalVariables

# def loss(model, x, y):
#    yPrediction = model(x)
#    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=yPrediction)

# def grad(model, inputs, targets):
#   with tf.GradientTape() as tape:
#     loss_value = loss(model, inputs, targets)
#   return loss_value, tape.gradient(loss_value, model.trainable_variables)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# global_step = tf.train.get_or_create_global_step()
# loss_value, grads = grad(model, features, labels)
#train = 

#python train.py
TRAINING_IMAGES_DIRECTORY = "./trainImages/"
TEST_IMAGES_DIRECTORY = "./testImages/"
IMDB_CSV_FILE_PATH = "./movie-genre-from-its-poster/MovieGenre.csv"
BATCH_LENGTH = GlobalVariables.BATCH_LENGTH
MAX_TRAINING_LENGTH = 2000
#MAX_TEST_LENGTH = 200
EPOCHS = 2
SAVED_SESSION_FILENAME = GlobalVariables.SAVED_SESSION_FILENAME
possibleGenres = GlobalVariables.possibleGenres

def getIndexOfGenre(genreString):
   for index, possibleGenreString in enumerate(possibleGenres):
      if possibleGenreString == genreString:
         return index

def getImages(imagesDirectory, csvFileDataRows, length, startingAtIndex=None):
   imageDataList = []
   imageGenresList = []
   fileList = os.listdir(imagesDirectory)
   i = 0
   if startingAtIndex != None:
      fileList = fileList[startingAtIndex:]
      i = startingAtIndex
   for imageFileName in fileList: 
      if len(imageDataList) < length:
         i += 1
         indexOfImageString = imageFileName.split(".")[0]
         indexOfImage = None
         if len(indexOfImageString) > 0:
            indexOfImage = int(indexOfImageString)
         else:
            continue
         genreStringTokens = csvFileDataRows[indexOfImage][4].split("|")
         genresForThisPoster = []
         for genreStringToken in genreStringTokens:
            #print genreStringToken
            if genreStringToken in possibleGenres:
               genresForThisPoster.append(genreStringToken)
         if len(genresForThisPoster) > 0:
            #print imagesDirectory + imageFileName
            imageData = cv2.imread(imagesDirectory + imageFileName)
            if imageData is not None:
               imageData = cv2.resize(imageData.astype(float), (227, 227)) #resize
               imageData = imageData.reshape((227, 227, 3))
               genreIndex = getIndexOfGenre(genresForThisPoster[0])
               correctVector = [0] * len(possibleGenres)
               correctVector[genreIndex] = 1
               imageGenresList.append(correctVector) #just use the first genre in the genresForThisPoster list as the correct genre
               imageDataList.append(imageData)  
      else:
         break      
   return imageDataList, imageGenresList, i


def savedSessionFileExists():
   for fileName in os.listdir("."):
      if SAVED_SESSION_FILENAME in fileName:
         return True
   return False

dropoutPro = GlobalVariables.dropoutPro

csvFileDataRows = utilities.parseImdbCSVFile(IMDB_CSV_FILE_PATH)
test_batch_x, test_batch_y, test_batch_index = getImages(TEST_IMAGES_DIRECTORY, csvFileDataRows, BATCH_LENGTH)
#x = tf.placeholder("float", [None, 227, 227, 3])
x = tf.placeholder("float", [BATCH_LENGTH, 227, 227, 3])
model = AlexNet.AlexNet(x, dropoutPro, len(possibleGenres))
y = tf.placeholder(tf.float32, [None, len(possibleGenres)])
pred = model.fc3
y_ = tf.nn.softmax(pred) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
   if savedSessionFileExists():
      print "Session Restored!"
      saver.restore(sess, SAVED_SESSION_FILENAME)
   else:
      sess.run(tf.global_variables_initializer())
   for i in range(EPOCHS):
       batch_index = 0
       iterations = MAX_TRAINING_LENGTH/BATCH_LENGTH
       print iterations
       for i in range(iterations):
          batch_x, batch_y, batch_index = getImages(TRAINING_IMAGES_DIRECTORY, csvFileDataRows, BATCH_LENGTH, batch_index)
          print batch_index
          _, c = sess.run([optimiser, cost], feed_dict={x: batch_x, y: batch_y})
          print "Cost: " + str(c)
       test_acc = sess.run(accuracy, feed_dict={x: test_batch_x, y: test_batch_y})

       
       print "Accuracy: " + str(test_acc)
   save_path = saver.save(sess, "./" + SAVED_SESSION_FILENAME)


