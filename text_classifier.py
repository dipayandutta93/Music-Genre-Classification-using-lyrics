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
import sys
from sklearn.pipeline import Pipeline
import pickle
from sklearn.externals import joblib
from sklearn import preprocessing

def pre_process():

	print "Processing the data..."	
	df = pd.read_csv('lyrics_final.csv')
	df.replace('?', -9999999, inplace=True)

	df.drop(['index'],1, inplace=True)
	
	dict_genres={'Rock':1, 'Country':2, 'Hip-Hop':3, 'Pop':4, 'Jazz':5}

	labels = []
	text = []

	for index, row in df.iterrows():
		labels.append(row[0])
		words = row[1].split()
		text.append(row[1].decode('utf-8', errors='ignore').encode('utf-8'))
		#text.append(words)

	#stemming
	text = [[stem(word) for word in sentence.split(" ")] for sentence in text]
	text = [" ".join(sentence) for sentence in text]

	print "Data processing done!!"

	return text,labels

def train(input):

	f = open("multiclass.txt", "w")

	text, labels = pre_process()
	
	print "Total songs: %s"%(len(labels))
	countVec = TfidfVectorizer(stop_words = 'english', sublinear_tf=True)

	#countVec = CountVectorizer(stop_words = 'english')

	x_train, x_test, y_train, y_test = cross_validation.train_test_split(text,labels,test_size=0.2)

	#Creating tf-idf vector for the documents

	x_trainCV = countVec.fit_transform(x_train)
	joblib.dump(countVec, "tfidf_vectorizer.pickle")

	x_testCV = countVec.transform(x_test)

	#converting the vectors into array to use further
	x_train = x_trainCV.toarray()
	x_test = x_testCV.toarray()

	x_train = preprocessing.normalize(x_train)
	x_train = preprocessing.scale(x_train)
	
	x_test = preprocessing.normalize(x_test)
	x_test = preprocessing.scale(x_test)
	
	joblib.dump(countVec, "tfidf_vectorizer.pickle")

	print "x_train: %s, x_test: %s, y_train: %s, y_test: %s"%(len(x_train),len(x_test),len(y_train),len(y_test))	
	
	if input == "svm":
		print "SVM classifier"
		svm = Pipeline([('vect', TfidfVectorizer(analyzer=lambda x: x)),('clf', SVC(kernel = 'linear'),)])
		svm = svm.fit(x_train,y_train)
		accuracy = svm.score(x_test,y_test)
		print "SVM: accuracy for tf-idf %s"%(accuracy)
	
	if input == "mnb":
		print "Multinomial Naive Bayes Classifier"
		mnb = MultinomialNB()
		mnb.fit(x_train,y_train)
		accuracy = mnb.score(x_test,y_test)
		print "accuracy for multinomial naive bayes: %s"%(accuracy)
		#print (confusion_matrix(y_train, mnb_predicted))
		f.write("accuracy for multinomial naive bayes: %s \n"%(accuracy))
	
	if input == "bnb":
		print "Bernoulli Naive Bayes Classifier"
		bnb = BernoulliNB()
		bnb.fit(x_train,y_train)
		accuracy = bnb.score(x_test,y_test)
		print "accuracy for bernoulli naive bayes: %s"%(accuracy)
		f.write("accuracy for bernoulli naive bayes: %s \n"%(accuracy))
	
	if input == "lr":
		print "Logistic Regression Classifier"
		lr = LogisticRegression(solver="liblinear", multi_class="ovr")
		lr.fit(x_train,y_train)
		print "accuracy for LogisticRegression: %s"%(lr.score(x_test,y_test))
		f.write("accuracy for logistic regression: %s \n"%(accuracy))
	
	if input == "dt":
		print "Decision Tree Classifier"
		dt = DecisionTreeClassifier()
		dt.fit(x_train,y_train)
		print "accuracy for Decision Tree: %s"%(dt.score(x_test,y_test))
		f.write("accuracy for decision tree: %s \n"%(accuracy))
	
	if input == "mlp":
		print "Multi Layer Perceptron Classifier"
		#Training and Testing on SCikit Neural Network library
		neural = MLPClassifier()
		neural.fit(x_train,y_train)

		joblib.dump(neural, "classifier.pickle")

		accuracy = neural.score(x_test, y_test)
		print "accuracy for Neural Network: %s"%(accuracy)
		
	if input == "rf":
		print "Random Forest Classifier"
		rf = RandomForestClassifier(n_estimators=100,max_features="sqrt").fit(x_train,y_train)
		joblib.dump(rf, "classifier.pickle")
		accuracy = rf.score(x_test, y_test)
		
		print "accuracy for Random Forest: %s"%(accuracy)
		f.write("accuracy for Random Forest: %s \n"%(accuracy))
	
	f.close()

def test(input_string):

	vectorizer = joblib.load("tfidf_vectorizer.pickle")
	classifier = joblib.load("classifier.pickle")
	
	tr = vectorizer.transform(input_string)

	predictions = classifier.predict(tr)
	print predictions[0]
	return predictions[0]
	
if __name__ == '__main__':

	input_param = sys.argv[1]
	train(input_param)
	