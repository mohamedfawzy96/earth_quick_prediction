import numpy as np
from preprocessing import preprocess
from data import get_dataset
from sklearn.metrics import r2_score

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

def relu(x, deriv=False):
        if deriv:
            return np.ones_like(x) * (x > 0)

        return x * (x > 0)

# Get and preprocess the data
X, Y = get_dataset()
X_train, X_test, y_train, y_test = preprocess(X,Y)
y_train = y_train.reshape(len(y_train),1)
y_test = y_test.reshape(len(y_test),1)

# Initialize weights
np.random.seed(1)
num_features = len(X_train[0])
w0 = 2*np.random.random((num_features, num_features + 1)) - 1
w1 = 2*np.random.random((num_features + 1, num_features + 2)) - 1
w2 = 2*np.random.random((num_features + 2, 1)) - 1

# Train model
learning_rate = 0.00001
print "Starting this may take a while..."
for i in range(100000):
	l0 = X_train 
	l1 = nonlin(np.dot(l0, w0))
	l2 = nonlin(np.dot(l1, w1))
	l3 = relu(np.dot(l2, w2))
	l3_error = y_train - l3
	l3_delta = l3_error * relu(l3, deriv=True) * learning_rate

	l2_error = l3_delta.dot(w2.T)
	l2_delta = l2_error * nonlin(l2, deriv=True)

	l1_error = l2_delta.dot(w1.T)
	l1_delta = l1_error * nonlin(l1, deriv=True)

	if ( i % 4999) == 0:
		print "Error:" + str(np.mean(np.abs(l3_error)))

	w2 += l2.T.dot(l3_delta)	
	w1 += l1.T.dot(l2_delta)
	w0 += l0.T.dot(l1_delta)

def predict(X1):
	l00 = X1
	l11 = nonlin(np.dot(l00, w0))
	l22 = nonlin(np.dot(l11, w1))
	l33 = relu(np.dot(l22, w2))
	return l33

error = y_test - predict(X_test)
print np.mean(np.abs(error))

