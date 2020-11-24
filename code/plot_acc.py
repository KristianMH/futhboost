import numpy
import futhark_data
import matplotlib.pyplot as plt



fp = open("auc_score_boolboost.txt", "r")
gen = futhark_data.loads(fp.readlines()[0].strip())
a = 0
for x in gen:
    a = x

b = numpy.loadtxt("auc_xgboost.txt")
plt.plot(a, label="futhboost")
plt.plot(b, label="xgboost")
plt.legend()
plt.show()

fp = open("reg_higgs_futh.txt", "r")
gen = futhark_data.loads(fp.readlines()[0].strip())
a = 0
for x in gen:
    a = x

b = numpy.loadtxt("reg_xgboost.txt")
plt.plot(a, label="futhboost")
plt.plot(b, label="xgboost")
plt.legend()
plt.show()
