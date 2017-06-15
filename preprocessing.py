from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# scaled the data to have a mean around 0
def sacler(X_train, X_test):
	 min_max_scaler = MinMaxScaler()
	 min_max_scaler.fit(X_train)
	 X_train = min_max_scaler.transform(X_train)
	 X_test = min_max_scaler.transform(X_test)

	 return X_train, X_test

# split the data to training and testing sets
def split_train_test(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test

# returning preprocossed data
def preprocess(X,Y):
	pca = PCA(n_components=10)
	X_train, X_test, y_train, y_test = split_train_test(X, Y)
	pca.fit(X_train)
	X_train, X_test = sacler(X_train, X_test)
	return X_train, X_test, y_train, y_test
