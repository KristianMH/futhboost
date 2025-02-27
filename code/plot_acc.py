import numpy
import futhark_data
import matplotlib.pyplot as plt



fp = open("500iter-auc.txt", "r")
gen = futhark_data.loads(fp.readlines()[0].strip())
a = 0
for x in gen:
    a = x

b = numpy.loadtxt("auc_500_1mill.txt")
plt.plot(a, label="futhboost")
plt.plot(b, label="xgboost")
plt.legend()
plt.title("AUC score between futhboost and xgboost on higgs data 5M")
plt.savefig("auc-higgs-1m.png")
plt.clf()

fp = open("reg_higgs_futh.txt", "r")
gen = futhark_data.loads(fp.readlines()[0].strip())
a = 0
for x in gen:
    a = x

b = numpy.loadtxt("reg_xgboost.txt")
plt.plot(a, label="futhboost")
plt.plot(b, label="xgboost")
plt.legend()
plt.title("rmse score between futhboost and xgboost on higgs data 5M")
plt.savefig("rmse-higgs.png")
