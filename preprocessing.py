from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def sacler(X):
	 min_max_scaler = preprocessing.MinMaxScaler()
	 X = min_max_scaler.fit_transform(X)
	 return X

def split_train_test(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
	return X_train, X_test, y_train, y_test

def preprocess(X,Y):
	X = sacler(X)
	X_train, X_test, y_train, y_test = split_train_test(X, Y)
	return X_train, X_test, y_train, y_test