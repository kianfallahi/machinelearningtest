import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def file(file_name):
	df = pd.read_csv(file_name)
	df = df.drop('Unnamed: 0', axis=1)
	df = df.drop('Name of episode', axis=1) # too much nan in this column, cant use fillna()
	df = df.dropna()

	vectorizerE = CountVectorizer() # for converting text to vector for ML model input
	vectorizerN = CountVectorizer()	# text columns : Episode, Name of show

	E = vectorizerE.fit_transform(df['Episode'])
	df['Episode'] = E.todense()
	N = vectorizerN.fit_transform(df['Name of show'])
	df['Name of show'] = N.todense()

	# replacing words with index
	d_station = df['Station'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['Station'] = df['Station'].map(d)

	d_station = df['Channel Type'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['Channel Type'] = df['Channel Type'].map(d)

	d_station = df['Season'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['Season'] = df['Season'].map(d)

	d_station = df['Day of week'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['Day of week'] = df['Day of week'].map(d)

	d_station = df['Genre'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['Genre'] = df['Genre'].map(d)

	d_station = df['First time or rerun'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['First time or rerun'] = df['First time or rerun'].map(d)

	d_station = df['# of episode in the season'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['# of episode in the season'] = df['# of episode in the season'].map(d)

	d_station = df['Movie?'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['Movie?'] = df['Movie?'].map(d)

	d_station = df['Game of the Canadiens during episode?'].value_counts().index
	i_station = []
	for i in range(len(d_station)):
	    i_station.append(i)
	d = dict(zip(d_station, i_station))
	df['Game of the Canadiens during episode?'] = df['Game of the Canadiens during episode?'].map(d)

	# removing :- from dates
	df['Date'].replace(regex=True,inplace=True,to_replace=r'-',value=r'')
	df['Start_time'].replace(regex=True,inplace=True,to_replace=r'-',value=r'')
	df['End_time'].replace(regex=True,inplace=True,to_replace=r'-',value=r'')

	df['Start_time'].replace(regex=True,inplace=True,to_replace=r':',value=r'')
	df['End_time'].replace(regex=True,inplace=True,to_replace=r':',value=r'')

	df['Start_time'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
	df['End_time'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')

	df = df.dropna()

	if file_name == 'data.csv':
		y = df['Market Share_total']
		x = df.drop('Market Share_total', axis=1)

		min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)) # scaling data between 0 and 1
		x = min_max_scaler.fit_transform(x)

		x,y = np.array(x), np.array(y)
		
		return (x,y)

	elif file_name == 'test.csv':
		x = df

		min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)) # scaling data between 0 and 1
		x = min_max_scaler.fit_transform(x)

		x = np.array(x)

		return (x)


data_file = 'data.csv'
test_file = 'test.csv'

x_train, y_train = file('data.csv')
x_test = file('test.csv')



from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
import tensorflow as tf

model = Sequential()
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam',loss='mae')

with tf.device('/GPU:0'):
    model.fit(x=x_train,
              y=y_train,
              batch_size=10000,
              epochs=100)


y_pred = model.predict(x_test)
print(y_pred)