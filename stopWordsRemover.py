import nltk
import string
import os
from nltk.corpus import stopwords

for filename in os.listdir(os.getcwd()):
	if(filename.startswith('msgsui')):
		with open(filename,'r', encoding='utf-8', errors='ignore') as inFile, open('tempOutFile.txt','w') as outFile:	
			for line in inFile.readlines():
			    print(" ".join([word for word in line.lower().translate(str.maketrans('', '', string.punctuation)).split() 
			    	if len(word) >=4 and word not in stopwords.words('english')]), file=outFile)

		inFile.close()
		outFile.close()
		with open('tempOutFile.txt','r') as inFile, open(filename,'r+') as outFile:
			outFile.truncate()
			outFile.write(inFile.read())
