import pandas as pd 
import numpy as np
from sklearn import preprocessing

#read data from csv file
dataframe = pd.read_csv('database.csv')
empty_features_list = ["Depth Error",
				      "Depth Seismic Stations", 
				      "Horizontal Error",
				      "Root Mean Square",
				      "ID",
				      "Azimuthal Gap",
				      "Horizontal Distance",
				      "Magnitude Error",
				      "Magnitude Seismic Stations"]

#remove features with many NaN values
def remove_empty_features(dataframe, list):
	dataframe = dataframe.drop(list, axis=1)
	dataframe = dataframe.dropna()
	return dataframe

#Encoding categorical features 
def change_categories_into_numbers(dataframe):
	categorical_features = ["Magnitude Type", "Source",
							"Location Source", "Magnitude Source",
							"Type","Status"
							]
	for name in categorical_features:
		dataframe[name] = pd.factorize(dataframe.loc[:,name])[0]
	return dataframe
	
#seperating each data and time in a colume (year, month, day, hour, minut, second)
def change_date_format(dataframe):
	dataframe['Year'] = pd.DatetimeIndex(dataframe['Date']).year
	dataframe['Month'] = pd.DatetimeIndex(dataframe['Date']).month
	dataframe['Day'] = pd.DatetimeIndex(dataframe['Date']).day
	dataframe['hour'] = pd.DatetimeIndex(dataframe['Time']).hour
	dataframe['minute'] = pd.DatetimeIndex(dataframe['Time']).minute
	dataframe['second'] = pd.DatetimeIndex(dataframe['Time']).second
	dataframe = dataframe.drop(["Date", "Time"], axis=1)
	return dataframe

#return the improved dataset
def get_dataset():
	dataframe1 = remove_empty_features(dataframe, empty_features_list)
	dataframe1 = change_categories_into_numbers(dataframe1)
	dataframe1 = change_date_format(dataframe1)
	Y = np.array(dataframe1.loc[:,"Magnitude"])
	X = np.array(dataframe1.drop(["Magnitude"], axis=1))
	return X, Y
