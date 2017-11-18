import csv
import pandas as pd

#Cleaning up the data
with open('lyrics.csv', 'rb') as inp, open('lyrics_out.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4] != "Alkebulan" and row[4] != "Other" and row[4] != "" and row[4] != "Not Available" and row[4] != "zora sourit" and row[5] != "":
            writer.writerow(row)

df = pd.read_csv('lyrics_out.csv')

#add a new column with word count of the lyrics of a song
df['word_count'] = df['lyrics'].str.split().str.len()

#remove rows with lyrics count less than 100

df = df[df['word_count'] > 100]

df = df.groupby('genre').head(1000)
#replace carriage returns
df = df.replace({'\n': ' '}, regex=True)

#convert all lyrics to lowercase and remove punctuations
df["lyrics"] = df['lyrics'].str.lower().replace('[^\w\s]','')

df.info()
df.to_csv('lyrics_final.csv')


