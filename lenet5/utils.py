import numpy as np
import struct
import pickle

# Define a function to fetch the dataset, stored in idx3-ubyte format
def read_idx_ubyte(file_path):
	with open(file_path, 'rb') as f: # When you open it in binary mode, it reads bit by bit
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def fetch_mnist_digit_data(folder_path):
	"""
	Loads the data of the MNIST dataset

	INPUT:

	folder_path : The path of the folder where the files reside. Left None if they are on the same folder as the caller.

	OUTPUT:

	X_train, X_test, t_train, t_test
	"""
	if folder_path is not None:
		folder_path = folder_path + "/"
	else:
		folder_path = ""

	train_feats_path = folder_path + "train-images.idx3-ubyte"
	test_feats_path = folder_path + "t10k-images.idx3-ubyte"
	train_target_path = folder_path + "train-labels.idx1-ubyte"
	test_target_path = folder_path + "t10k-labels.idx1-ubyte"

	X_train = read_idx_ubyte(train_feats_path)
	X_test = read_idx_ubyte(test_feats_path)
	t_train = read_idx_ubyte(train_target_path)
	t_test = read_idx_ubyte(test_target_path)

	# Add additional dimension so we have 4D tensors (expected by Conv2D operation in TF)
	X_train = X_train.reshape(*X_train.shape, 1)
	X_test = X_test.reshape(*X_test.shape, 1)
	return X_train, X_test, t_train, t_test


def fetch_cifar10_batch(file_name):
    """
    Returns tuple X, t, where X is a np.array with the training data and t is the corresponding labels vector.
    """
    with open(file_name, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        # Extract data and labels
        data = data_dict[b'data']
        labels = data_dict[b'labels']

        # Reshape data to a 4D tensor (observations,H,W,C)
    X = np.reshape(data, [data.shape[0],32,32,3], order='F')
    t = np.array(labels)
        
    for i in range(X.shape[0]):
        for c in range(X.shape[3]):
            X[i,:,:,c] = X[i,:,:,c].T
            
    return X, t


# Define a function to OneHotEncode the labels
def one_hot_encode(t):
	"""
	Returns a matrix (2D np.ndarray) with the one_hot_encoded version of labels vector t
	"""
	K = np.max(t) + 1 # Determine the number of classes
	N = t.shape[0]
	T = np.zeros((N, K)) # Output matrix
	T[np.arange(N), t] = 1
	T = T.astype(np.uint8)
	return T


