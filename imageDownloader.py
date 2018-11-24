import tensorflow as ts
import sys
import urllib
import utilities

labels = ["Animation", "Adventure", "Comedy", "Action", "Family", "Romance", "Drama", "Crime", "Thriller",
   "Fantasy", "Horror", "Biography", "History", "Mystery", "Sci-Fi", "War", "Sport", "Music", "Documentary",
   "Musical", "Western", "News", "Adult"]

#Goes over dataRows and downloads each image and put it all in image directory
def downloadImagesIntoImagesDirectory(dataRows):
   images = []
  
   for dataRowIndex, dataRow in enumerate(dataRows):
      imageURLIndex = -1
      for index, string in enumerate(dataRow):
         if "http" in string and "/title/" not in string:
            imageURLIndex = index
            break
      if imageURLIndex != -1:
	      imageToDownloadURL = dataRow[imageURLIndex]  
	      if len(dataRow[imageURLIndex].strip().split("\"")) > 1:
	      	imageToDownloadURL = dataRow[imageURLIndex].strip().split("\"")[1]
	      if "http" in imageToDownloadURL:
	          f = open("./images/" + str(dataRowIndex) + ".jpg", "wb")
	          f.write(urllib.urlopen(imageToDownloadURL).read())
	          f.close()
	  

#Gets all possible labels
def getLabels(dataRows):
   labels = []
   for dataRow in dataRows:
      genres = dataRow[4]
      genresList = []
      if "." not in genres:
         genresList = genres.split("|")
      for genre in genresList:
         if genre not in labels:
            labels.append(genre)
   print labels
      

def main(imdbCSVFilePath):
   dataRows = utilities.parseImdbCSVFile(imdbCSVFilePath)
   #downloadImagesIntoImagesDirectory(dataRows)
   getLabels(dataRows)
   #print dataRows

#python main.py movie-genre-from-its-poster/MovieGenre.csv 
main(sys.argv[1])