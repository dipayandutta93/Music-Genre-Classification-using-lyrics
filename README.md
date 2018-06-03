# Music-Genre-Classification-using-lyrics

### Abstract 

This project aims to build a system that can identify the genre of a song based on its lyrics. We identify a set of features that establish the style of a particular song. We curate a set of songs with ve labels - Rock, Hip-Hop, Jazz, Country and Pop. Then we design three models to classify the songs into their genres - Multi Layer Perceptron for multiclass classification, Random Forest for binary classification and Convolutional Neural Networks with word embeddings. We provide a user interface which would enable a user to input the lyrics of a particular song and our program would predict its genre based on the content of the lyrics.

### Introduction

In the field of Natural Language Processing, the classification of genres of a song solely based on the lyrics is considered a challenging task. Because audio features of a song also provides valuable information to classify the song into its respective genre. Previously researchers have tried several approaches to this problem, but they have failed to identify a method which would perform significantly well in this case. SVM, KNN and Naive Bayes have been used previously in lyrical classification research. But, classification into more than 10 genres have not been particularly successful, because then the clear boundary between the genres is often lost. So, we try to use a dataset of five genres. Hence, we try to approach this problem as a supervised learning problem applying several methods. We analysed the relative advantages and disadvantages of each of the methods and finally reported our observations. With the advent of deep learning, it has been observed that Neural Networks perform better than the previously used models. So we designed a Convoluted Neural Network using glove word embeddings and analysed its performance.

### Dataset

The dataset for this problem was not abundant mostly due to copyright issues. However, after comparing datasets from several sources, we found out a data set in Kaggle which was most suited for our purpose. The dataset is basically a collection of 380000+ lyrics from songs scraped from metrolyrics.com.The structure of the data is index/song/year/artist/genre/lyrics. The data was not properly structured according to our needs like there were some songs without any genre classified to it or there were some songs whose lyrics were absent. Sowe had to process our data before it could be fitted to any model for classification. Initially, we had to remove some irrelevant data from our dataset, making it more compact and easy to access. Like we removed artist and song year information thus creating just lyrics and genre mapping in our dataset. Then we extracted songs of five genres - Rock, Hip-Hop, Pop, Country, Jazz. And extracted 5000 songs from each genre, making the dataset practical and easy to analyze. Then we removed some songs which had very few words in its lyrics. Lyrics also contained some rhyming schemes like [chorus], [verse], [x1],[x2], we removed them for simplicity. Then we tokenized the lyrics text using NLTK tool in Python. Further, we applied stemming and removed punctuations. For stemming we used Porter Stemmer as we found it to be very effective. We also did some pre-processing of data for each of our models, which would be explained later.

### Data Analysis

After preprocessing we analysed the data and identified the features of data which is the first step of any machine learning problem. We used Spark to analyse the data and visualized the data. This analysis helped us understand the features of the data that would be most useful for the task in our hand. We evaluated the average length of lyrics in each genre, and we had an interesting insight, Hip-Hop songs were longer as compared to the other genres. And the rest of the genres had almost similar lengths. Then we calculated the average number of unique words in each genre. (Figure 1) Here as well we found out that Hip-Hop songs had more unique words as compared to the rest. Then we calculated the most common words of each genre. (Figure 2) This would help us understand any correlation between the words used in lyrics and the genre type.

![image](https://user-images.githubusercontent.com/32987993/40891455-d79d4564-6753-11e8-8287-fac13dbf90e4.png)

### Approaches

We have taken three approaches to the problem, resulting in three models. In our first approach we use term frequency and inverse document frequency as our feature vectors and the genre classes as our labels to identify. We developed Naive Bayes, Random Forest, Support Vector machine and Multi Layer Perceptron model to classify the songs into multiple classes. In the second model we convert the problem into a binary classication problem and developed a classifier which will identify a song as Rock or Non Rock, Hip-Hop or Non Hip-Hop. We did this to identify the genres which are more distinguishable from the rest on the basis of the content of its songs. The third model that we used was the most effective of all, we used a Convolutional Neural Network, with Glove word embeddings as the feature vector.

#### Model I

We used term frequency and inverse document frequency as our feature vectors and the genre classes as our labels to identify.

##### Bag of Words

This is one of the most common approaches in text retrieval. Here, any unique term occurring in any of the
document of the collection is regarded as a feature. One simple approach is to count the frequency of the word in
the entire lyrical text. Another approach is term weighting scheme based on the importance of a term to describe and discriminate between documents, such as the popular tf - idf (term frequency times inverse document frequency) weighting scheme. In this model, a document is denoted by d, a term (token) by t, and the number of documents in a corpus by N. The term frequency tf(t, d) denotes the number of times term t appears in document d. The number of documents in the collection that term t occurs in is denoted as document frequency df(d). The tf-idf weight of a term in a document is computed as:
##### tf x idf(t, d) = tf (t, d) x ln(N/df(t))
We have also normalized the vector after applying the Count Vectorizer and Tf-Idf Weighing scheme.

##### Word2Vec

Next, we used the word vectors (word2vec) to represent our lyrical text. These semantic vectors preserve most of the relevant information in a text while having relatively low dimensionality. Word2Vec is an algorithm that takes every word in your vocabulary that is, the text that needs to be classied is turned into a unique vector that can be added, subtracted, and manipulated in other ways just like a vector in space. We trained word vectors using python's genism library. We generated 100-dimensional word2vec embedding trained on the benchmark data itself.

##### Algorithms

With our features and labels ready we fed them into a classier and trained it. We used 4:1 split of the dataset for training and testing. We used python's sci-kit learn library to implement the following algorithms: 

Naive Bayes: Implemented Bernoulli and Multinomial Naive Bayes, Support Vector Machine: Used the linear kernel, Logistic Regression, Decision Tree, Random Forest: Used 100 trees and the majority of all the classifications are the result, MultiLayer Perceptron Model: Experimented with various activation functions and hidden layers, Extra Trees Classifier: Used this algorithm to test with word2vec feature vectors, Extra Trees Classier: Used this algorithm to test with
word2vec feature vectors.

#### Model II

We converted the problem into a binary classification problem and developed a classifier which will identify a song as Rock or Non Rock, Hip-Hop or Non Hip-Hop. We did this to identify the genres which are more distinguishable from the rest on the basis of the content of its songs.

##### Features

We divided the data into two groups for each of the genre classes, like grouping dataset into rock and non-rock, hip- hop and non hip-hop etc. We used one hot encoding to represent the class labels and used term frequency-inverse document frequency to represent the features. We implemented this model to identify the genres which were easily classified as compared to the rest.

#### Model III

We used a Convolutional Neural Network to classify the songs into their respective genres. We used pre-trained glove vectors for this model.

##### Description of the model
The glove model we used is Google Glove 6B vector 100d. We have implemented two CNN models using Keras library:
i. Simple convolution model: We have implemented a single layer of convoluted and maxpool layer.
ii. Dense convolution model: We have implemented multiple convoluted and maxpool layers with filter sizes of 3, 4 and 5.(Figure 3)

![image](https://user-images.githubusercontent.com/32987993/40891825-88c3bdea-675a-11e8-91ca-80867925c45b.png)
