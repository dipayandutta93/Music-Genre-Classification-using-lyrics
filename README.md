# Music-Genre-Classification-using-lyrics

###Abstract 

This project aims to build a system that can identify the genre of a song based on its lyrics. We identify a set of features that establish the style of a particular song. We curate a set of songs with ve labels - Rock, Hip-Hop, Jazz, Country and Pop. Then we design three models to classify the songs into their genres - Multi Layer Perceptron for multiclass classification, Random Forest for binary classification and Convolutional Neural Networks with word embeddings. We provide a user interface which would enable a user to input the lyrics of a particular song and our program would predict its genre based on the content of the lyrics.

###Introduction

In the field of Natural Language Processing, the classification of genres of a song solely based on the lyrics is considered a challenging task. Because audio features of a song also provides valuable information to classify the song into its respective genre. Previously researchers have tried several approaches to this problem, but they have failed to identify a method which would perform significantly well in this case. SVM, KNN and Naive Bayes have been used previously in lyrical classification research. But, classification into more than 10 genres have not been particularly successful, because then the clear boundary between the genres is often lost. So, we try to use a dataset of five genres. Hence, we try to approach this problem as a supervised learning problem applying several methods. We analysed the relative advantages and disadvantages of each of the methods and finally reported our observations. With the advent of deep learning, it has been observed that Neural Networks perform better than the previously used models. So we designed a Convoluted Neural Network using glove word embeddings and analysed its performance.

###Dataset

The dataset for this problem was not abundant mostly due to copyright issues. However, after comparing datasets from several sources, we found out a data set in Kaggle which was most suited for our purpose. The dataset is basically a collection of 380000+ lyrics from songs scraped from metrolyrics.com.The structure of the data is index/song/year/artist/genre/lyrics. The data was not properly structured according to our needs like there were some songs without any genre classified to it or there were some songs whose lyrics were absent. Sowe had to process our data before it could be fitted to any model for classification. Initially, we had to remove some irrelevant data from our dataset, making it more compact and easy to access. Like we removed artist and song year information thus creating just lyrics and genre mapping in our dataset. Then we extracted songs of five genres - Rock, Hip-Hop, Pop, Country, Jazz. And extracted 5000 songs from each genre, making the dataset practical and easy to analyze. Then we removed some songs which had very few words in its lyrics. Lyrics also contained some rhyming schemes like [chorus], [verse], [x1],[x2], we removed them for simplicity. Then we tokenized the lyrics text using NLTK tool in Python. Further, we applied stemming and removed punctuations. For stemming we used Porter Stemmer as we found it to be very effective. We also did some pre-processing of data for each of our models, which would be explained later.

### Data Analysis

After preprocessing we analysed the data and identified the features of data which is the first step of any machine learning problem. We used Spark to analyse the data and visualized the data. This analysis helped us understand the features of the data that would be most useful for the task in our hand. We evaluated the average length of lyrics in each genre, and we had an interesting insight, Hip-Hop songs were longer as compared to the other genres. And the rest of the genres had almost similar lengths. Then we calculated the average number of unique words in each genre. (Figure 1) Here as well we found out that Hip-Hop songs had more unique words as compared to the rest. Then we calculated the most common words of each genre. (Figure 2) This would help us understand any correlation between the words used in lyrics and the genre type.

![image](https://user-images.githubusercontent.com/32987993/40891455-d79d4564-6753-11e8-8287-fac13dbf90e4.png)




