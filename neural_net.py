import numpy as np
from preprocessing import preprocess
from data import get_dataset

learning_rate = 0.000001

# Get and preprocess the data
X, Y = get_dataset()
X_train, X_test, y_train, y_test = preprocess(X,Y)
y_train = y_train.reshape(len(y_train),1)
y_test = y_train.reshape(len(y_train),1)


np.random.seed(1)

# Initialize weights
num_features = len(X_train[0])
w0 = 2*np.random.random((num_features, num_features + 1)) - 1
w1 = 2*np.random.random((num_features + 1, 1)) - 1

# Train model
print "Starting..."
for i in range(60000):
	l0 = X_train 
	l1 = np.dot(l0, w0)
	l2 = np.dot(l1, w1)
	l2_error = y_train - l2
	if ( i % 10000) == 0:
		print "Error:" + str(np.mean(np.abs(l2_error)))
	l2_delta = l2_error * learning_rate
	l1_error = l2_delta.dot(w1.T)
	l1_delta = l1_error * learning_rate
	w1 += l1.T.dot(l2_delta)
	w0 += l0.T.dot(l1_delta)



