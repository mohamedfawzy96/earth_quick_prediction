import numpy as np
from preprocessing import preprocess
from data import get_dataset
from sklearn.metrics import r2_score

learning_rate = 0.000001
learning_rate2 = 0.00001
learning_rate3 = 0.00001

# Get and preprocess the data
X, Y = get_dataset()
X_train, X_test, y_train, y_test = preprocess(X,Y)
y_train = y_train.reshape(len(y_train),1)
y_test = y_train.reshape(len(y_train),1)


np.random.seed(1)

# Initialize weights
num_features = len(X_train[0])
w0 = 2*np.random.random((num_features, num_features + 1)) - 1
w1 = 2*np.random.random((num_features + 1, num_features + 2)) - 1
w2 = 2*np.random.random((num_features + 2, 1)) - 1
# Train model
print "Starting..."
for i in range(60000):
	l0 = X_train 
	l1 = np.dot(l0, w0)
	l2 = np.dot(l1, w1)
	l3 = np.dot(l2, w2)
	l3_error = y_train - l3
	l3_delta = l3_error * learning_rate

	l2_error = l3_delta.dot(w2.T)
	l2_delta = l2_error * learning_rate2

	l1_error = l2_delta.dot(w1.T)
	l1_delta = l1_error * learning_rate3



	if ( i % 10000) == 0:
		print "Error:" + str(np.mean(np.abs(l3_error)))

	w2 += l2.T.dot(l3_delta)	
	w1 += l1.T.dot(l2_delta)
	w0 += l0.T.dot(l1_delta)

def predict(X):
	l0 = X_train 
	l1 = np.dot(l0, w0)
	l2 = np.dot(l1, w1)
	l3 = np.dot(l2, w2)
	return l3

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

error = y_test - predict(X_test)
print np.mean(np.abs(error))

