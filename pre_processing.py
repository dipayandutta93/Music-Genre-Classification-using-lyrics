import csv
import pandas as pd
from nltk.corpus import wordnet
from stemming.porter2 import stem

def data_cleanup():

	#Cleaning up the data
	with open('lyrics.csv', 'rb') as inp, open('lyrics_out.csv', 'wb') as out:
	    writer = csv.writer(out)
	    for row in csv.reader(inp):
	        if row[4] != "Alkebulan" and row[4] != "Other" and row[4] != "" and row[4] != "Not Available" and row[4] != "zora sourit" and row[5] != "":
	            writer.writerow(row)

def multi_class_data():
	
	data_cleanup()

	df = pd.read_csv('lyrics_out.csv')

	#add a new column with word count of the lyrics of a song
	df['word_count'] = df['lyrics'].str.split( ).str.len()

	df["lyrics"] = df['lyrics'].str.lower()

	df['lyrics'] = df['lyrics'].str.strip('[]')
	df['lyrics'] = df['lyrics'].str.strip('()')
	df["lyrics"] = df['lyrics'].str.replace('[^\w\s]','')
	df["lyrics"] = df['lyrics'].str.replace('chorus','')
	df["lyrics"] = df['lyrics'].str.replace(':','')
	df["lyrics"] = df['lyrics'].str.replace(',','')
	df["lyrics"] = df['lyrics'].str.replace('verse','')
	df["lyrics"] = df['lyrics'].str.replace('x1','')
	df["lyrics"] = df['lyrics'].str.replace('x2','')
	df["lyrics"] = df['lyrics'].str.replace('x3','')
	df["lyrics"] = df['lyrics'].str.replace('x4','')
	df["lyrics"] = df['lyrics'].str.replace('x5','')
	df["lyrics"] = df['lyrics'].str.replace('x6','')
	df["lyrics"] = df['lyrics'].str.replace('x7','')
	df["lyrics"] = df['lyrics'].str.replace('x8','')
	df["lyrics"] = df['lyrics'].str.replace('x9','')
	df["lyrics"] = df['lyrics'].str.decode('utf-8', errors='ignore')

	#remove rows with lyrics count less than 100

	#df = df[df['genre']!="Pop"]
	df = df[df['genre']!="Folk"]
	#df = df[df['genre']!="Jazz"]
	df = df[df['genre']!="R&B"]
	df = df[df['genre']!="Indie"]
	df = df[df['genre']!="Electronic"]
	df = df[df['genre']!="Metal"]

	df = df[df['word_count'] > 100]

	df = df.groupby('genre').head(1000)
	#replace carriage returns
	df = df.replace({'\n': ' '}, regex=True)

	#convert all lyrics to lowercase and remove punctuations
	df["lyrics"] = df['lyrics'].str.lower().replace('[^\w\s]','')

	del df['song'],df['year'],df['artist'],df['word_count']
	print (df.head())

	df.to_csv('lyrics_final.csv', index=False)

def genre(x, type):
	if type in x:
		return type
	else:
		return "Non %s"%(type) 


def binary_data(type):
	
	data_cleanup()
	print "type:%s"%(type)

	df = pd.read_csv('lyrics_out.csv')

	#add a new column with word count of the lyrics of a song
	df['word_count'] = df['lyrics'].str.split().str.len()
	
	df["genre values"] = df["genre"].apply(lambda x: genre(x, type))

	#remove rows with lyrics count less than 100
	df = df[df['word_count'] > 100]

	#df = df.groupby('genre').head(1000)
	#replace carriage returns
	df = df.replace({'\n': ' '}, regex=True)

	#convert all lyrics to lowercase and remove punctuations
	df["lyrics"] = df['lyrics'].str.lower().replace('[^\w\s]','')

	del df['song'],df['year'],df['artist'],df['word_count'],df['genre']

	df.to_csv("lyrics_%s.csv"%(type), index=False)

if __name__ == '__main__':

	#for multi class classification
	multi_class_data()

	#for binary classifier
	#genre_type = ["Rock", "Jazz", "Pop", "Country", "Hip-Hop"]

	#for type in genre_type:
	#	binary_data(type)



