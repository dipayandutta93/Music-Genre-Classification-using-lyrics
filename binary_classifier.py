import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import cross_validation, svm
from stemming.porter2 import stem
from sklearn.metrics import confusion_matrix

def binary_text_classifier(type):
	
	f = open("binaryclass.txt", "w")
	print "For type %s"%(type)
	f.write("For type %s"%(type))

	filename = "lyrics_%s.csv"%(type)
	print (filename)
	df = pd.read_csv(filename)
	df.replace('?', -9999999, inplace=True)

	df.drop(['index'],1, inplace=True)

	type1 = type
	type2 = "Non %s"%(type)

	dict_genres={type:1, "Non %s"%(type):2}
	labels = []
	text = []

	print (dict_genres)
	for index, row in df.iterrows():
		labels.append(dict_genres[row[1]])
		text.append(row[0].decode('utf-8', errors='ignore').encode('utf-8'))

	countVec = TfidfVectorizer(stop_words = 'english')

	x_train, x_test, y_train, y_test = cross_validation.train_test_split(text,labels,test_size=0.1)

	print "x_train: %s, x_test: %s, y_train: %s, y_test: %s"%(len(x_train),len(x_test),len(y_train),len(y_test))

	x_trainCV = countVec.fit_transform(x_train)
	x_testCV = countVec.transform(x_test)

	#converting the vectors into array to use further
	x_train = x_trainCV.toarray()
	x_test = x_testCV.toarray()

	#print "x_train: %s, x_test: %s, y_train: %s, y_test: %s"%(len(x_train),len(x_test),len(y_train),len(y_test))	
	
	
	mnb = MultinomialNB()
	mnb.fit(x_train,y_train)
	accuracy = mnb.score(x_test,y_test)
	print "accuracy for multinomial naive bayes: %s"%(accuracy)
	f.write("accuracy for multinomial naive bayes: %s \n"%(accuracy))

	gnb = BernoulliNB()
	gnb.fit(x_train,y_train)
	accuracy = gnb.score(x_test,y_test)
	print "accuracy for gaussian naive bayes: %s"%(accuracy)
	f.write("accuracy for bernoulli naive bayes: %s \n"%(accuracy))

	lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
	lr.fit(x_train,y_train)
	print "accuracy for LogisticRegression: %s"%(lr.score(x_test,y_test))
	f.write("accuracy for logistic regression: %s \n"%(accuracy))

	dt = DecisionTreeClassifier()
	dt.fit(x_train,y_train)
	print "accuracy for Decision Tree: %s"%(dt.score(x_test,y_test))
	f.write("accuracy for decision tree: %s \n"%(accuracy))
	
	#Training and Testing on SCikit Neural Network library
	neural = MLPClassifier().fit(x_train,y_train)
	accuracy = neural.score(x_test, y_test)
	print "accuracy for Neural Network: %s"%(accuracy)
	f.write("accuracy for Neural Network: %s \n"%(accuracy))
	
	rf = RandomForestClassifier().fit(x_train,y_train)
	accuracy = rf.score(x_test, y_test)
	print "accuracy for Random Forest: %s"%(accuracy)
	f.write("accuracy for Random Forest: %s \n"%(accuracy))
	f.close()
	

if __name__ == '__main__':

	genre_type = ["Rock", "Jazz", "Pop", "Country", "Hip-Hop"]

	for type in genre_type:
		binary_text_classifier(type)
	print "Accuracy for the classifier is %r"%(accuracy)	