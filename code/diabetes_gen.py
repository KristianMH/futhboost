import numpy
import futhark_data
import sklearn.datasets as dat

fileHandler = open("data/diabetes_bin","wb")
d = dat.load_diabetes()
data = d.data.astype("float32")
target = d.target.astype("float32")
futhark_data.dump(data, fileHandler, True)
futhark_data.dump(target, fileHandler, True)
#fileHandler.write("let woopdata = "+futhark_data.dumps(data))
#fileHandler.write("let wooptarget = "+futhark_data.dumps(target))
