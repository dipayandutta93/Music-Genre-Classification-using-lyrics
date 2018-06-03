# Music-Genre-Classification-using-lyrics

Abstract 

This project aims to build a system that can identify the genre of a song based on its lyrics. We identify a set of features that establish the style of a particular song. We curate a set of songs with ve labels - Rock, Hip-Hop, Jazz, Country and Pop. Then we design three models to classify the songs into their genres - Multi Layer Perceptron for multiclass classification, Random Forest for binary classification and Convolutional Neural Networks with word embeddings. We provide a user interface which would enable a user to input the lyrics of a particular song and our program would predict its genre based on the content of the lyrics.

Introduction

In the field of Natural Language Processing, the classification of genres of a song solely based on the lyrics is considered a challenging task. Because audio features of a song also provides valuable information to classify the song into its respective genre. Previously researchers have tried several approaches to this problem, but they have failed to identify a method which would perform significantly well in this case. SVM, KNN and Naive Bayes have been used previously in lyrical classification research. But, classification into more than 10 genres have not been particularly successful, because then the clear boundary between the genres is often lost. So, we try to use a dataset of five genres. Hence, we try to approach this problem as a supervised learning problem applying several methods. We analysed the relative advantages and disadvantages of each of the methods and finally reported our observations. With the advent of deep learning, it has been observed that Neural Networks perform better than the previously used models. So we designed a Convoluted Neural Network using glove word embeddings and analysed its performance.

