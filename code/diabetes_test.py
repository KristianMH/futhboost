import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets


d = sklearn.datasets.load_diabetes()

data = d.data.astype("float32")
target = d.target.astype("float32")
xgboostdatat = xgb.DMatrix(data, label=target)
xgboostdata = xgb.DMatrix(data, label=target)
param = {"objective":"reg:squarederror", "max_depth":1, "eta":0.3, "reg_lambda":0.5,
         "tree_method":"exact"}#, "max_bin":10}
num_round=1
#progress = dict()
watchlist  = [(xgboostdatat, "val-rmse"), (xgboostdata,'train-rmse')]
bst = xgb.train(param, xgboostdata, num_round, watchlist)
print(xgboostdata.feature_names)
#print(bst.get_split_value_histogram("f8"))
#print(bst.get_score())
#ha = xgb.Booster(param)
#print(ha.boost(xgboostdata, ))
#print(bst.predict(trai))
#print(progress)
#print(bst.get_dump())
# ha = bst.predict(xgboostdata)
# unique, counts = np.unique(ha, return_counts=True)
# print(counts)

for i in range(num_round):
   xgb.plot_tree(bst, num_trees=i)
   plt.show()
#     #bst.dump_model("model.b")
#     #bst.tree()
ha = data[:, 8] < -3.3761e-3
print(np.logical_not(ha).sum())
#print (data[ha][8].sum())
#ha = str(ha).replace("F","f")
#ha = ha.replace("T","t")
#ha = ha.replace(" ", ",")
#ha = ha.replace(",,", ",")
#print("let bool_arr = "+ha)
#print(ha.sum())
