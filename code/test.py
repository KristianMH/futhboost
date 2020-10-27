import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("data.txt")
train = xgb.DMatrix(data[:,:2], label=data[:,2])
a = np.random.rand(10,3)*[10,10,1]
a[:,2] = 3*a[:,0]+2*a[:,1]+np.random.rand(1,10)*2
test = xgb.DMatrix(a[:,:2], label=a[:,2])
param = {"objective":"reg:squarederror", "max_depth":3, "eta":0.3, "reg_lambda":0.5,
         "tree_method":"hist", "max_bin":3}
num_round=1
#progress = dict()
watchlist  = [(test, "val-rmse"), (train,'train-rmse')]
bst = xgb.train(param, train, num_round, watchlist)
print(bst.predict(train))
#print(progress)
#print(bst.get_dump())
for i in range(num_round):
    xgb.plot_tree(bst, num_trees=i)
    plt.show()
#bst.dump_model("model.b")
#bst.tree()

