import csv
import pandas as pd

df = pd.read_csv('lyrics_final.csv')

for index, row in df.iterrows():
	if row[1] == "Pop":
		with open("genres_data/pop/%s"%(row[0]), "a") as pop_file:
			pop_file.write(row[2])

	if row[1] == "Rock":
		with open("genres_data/rock/%s"%(row[0]), "a") as rock_file:
			rock_file.write(row[2])

	if row[1] == "Jazz":
		with open("genres_data/jazz/%s"%(row[0]), "a") as jazz_file:
			jazz_file.write(row[2])

	if row[1] == "Country":
		with open("genres_data/country/%s"%(row[0]), "a") as country_file:
			country_file.write(row[2])

	if row[1] == "Hip-Hop":
		with open("genres_data/hiphop/%s"%(row[0]), "a") as hiphop_file:
			hiphop_file.write(row[2])
