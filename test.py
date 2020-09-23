import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("data.txt")
train = xgb.DMatrix(data[:,:2], label=data[:,2])
#test  = xgb.DMatrix(data[:,2].reshape(10,1))
param = {"objective":"reg:squarederror", "max_depth":1}
num_round=1
bst = xgb.train(param, train, num_round)
#xgb.plot_tree(bst)
#plt.show()
#bst.dump_model("model.b")
bst.tree()
