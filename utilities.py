
#Returns an array of rows of csv file. 
def parseImdbCSVFile(filePath):
   imdbCSVFile = open(filePath, "r")
   lines = imdbCSVFile.readlines()
   dataRows = []
   for line in lines[1:]:
      dataRows.append(line.strip().split(","))
   imdbCSVFile.close()
   return dataRows