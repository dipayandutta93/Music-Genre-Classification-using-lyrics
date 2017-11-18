import findspark

findspark.init(spark_home=r'spark-2.2.0-bin-hadoop2.7')

from pyspark import SparkConf, SparkContext, SQLContext
from operator import add
from pyspark.mllib.feature import HashingTF, IDF
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import string
from pyspark.sql.functions import length

conf = SparkConf()
sc = SparkContext(conf=conf)

def remove_Stop_words(word_list):

    for word in word_list:
        if word not in nltk.corpus.stopwords.words('english') and word not in string.punctuation:
            word_list.append(word)
    return word_list

#read the contents of csv file
data = sc.textFile(r'lyrics_final.csv')
sqlContext = SQLContext(sc)
df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true")\
    .load(r'lyrics_final.csv')
df.printSchema()

#get all distinct genres in the csv file
list_of_genres=df.select('Genre').distinct().rdd.map(lambda r: r[0]).collect()
all_genres=[i.Genre for i in df.select('Genre').distinct().collect()]

f = open("out.txt", "w")

stopwords = set(stopwords.words('english'))
average_word_list=[]
unique_word_list=[]
stop_word_list = []

for genres in all_genres:

    top_word_list = []
    subset_df = df.filter(df.genre == genres)

    number_of_songs = subset_df.count()

    #Count the total number of words for each genre
    set_of_words=subset_df.select("lyrics").rdd.flatMap(lambda x: x[0].split(" "))
    total_no_of_words=set_of_words.count()

    #Average number of words per genre
    average_words=total_no_of_words/number_of_songs
    average_word_list.append(average_words)

    #Identifying the number of words appearing for each genre
    count_of_each_words=set_of_words.map(lambda x: (x, 1)).reduceByKey(add)
    total_unique_words = count_of_each_words.count() / number_of_songs
    unique_word_list.append(total_unique_words)

    #Determine the count ofeach of the words
    top_ten_wordcount = count_of_each_words.takeOrdered(10, key=lambda x: -x[1])
    #after_removed=remove_Stop_words(set_of_words)
    #print("skdbv jk",after_removed)
    stopwords_removed_top_ten = set_of_words.filter(lambda x: x not in stopwords).map(lambda x: (x, 1)).reduceByKey(\
        add).takeOrdered(12, key=lambda x: -x[1])

    #Write the output in a file
    f.write("For genre "+ genres+ "\n" +"Total number of songs: "+ str(number_of_songs) +"\n"+ "Total words in the song: "+\
          str(total_no_of_words) + "\n" +"Average number of words per song: "+str(average_words) + "\n"+"Total number of unique words per song: "+\
          str(total_unique_words)+ "\n"+"Top ten most frequent word: "+ str(top_ten_wordcount)+ "\n"+"Without stopwords top 10 words:"+\
            str(stopwords_removed_top_ten)+"\n" +"-------------------------------------------"+"\n")

    #computing TF-IDF
    hashingTF = HashingTF()
    tf = hashingTF.transform(set_of_words)

    # While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
    # First to compute the IDF vector and second to scale the term frequencies by IDF.
    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)

    idfIgnore = IDF(minDocFreq=2).fit(tf)
    tfidfIgnore = idfIgnore.transform(tf)

    sumrdd = tfidf.map(lambda v: v.values.sum())

f.close()

#visualizations

#Generate the plot for average number of words used
y_pos = np.arange(len(all_genres))

plt.bar(y_pos, average_word_list, align='center', alpha=0.5)
plt.xticks(y_pos, all_genres)
plt.ylabel('Average number of words used')
plt.title('Genres')

plt.show()

#Generate the plot for average number of words used
plt.bar(y_pos, unique_word_list, align='center', alpha=0.5)
plt.xticks(y_pos, all_genres)
plt.ylabel('Number of unique words used')
plt.title('Genres')

plt.show()

