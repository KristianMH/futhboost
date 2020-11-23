import numpy as np
import futhark_data
import pandas as pd
#from sklearn.datasets import dump_svmlight_file

test_size=5*10**6


data = pd.read_csv("data/HIGGS.csv", delimiter=",", header=None)
data = data.to_numpy()[:test_size]
data = np.where(data==-999.0, np.nan, data)
#print(data = -999.0)
#(l, c) = data.shape
#train = data[:, :l-test]
print(data.shape)
target = data[:,0].astype("float32")
print(target.shape)
data = data[:,1:].astype("float32")
print(data.shape)
fileHandler = open("data/HIGGS_training", "wb")
futhark_data.dump(data, fileHandler, True)
futhark_data.dump(target, fileHandler, True)
