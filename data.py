import numpy as np

a = np.random.rand(10,3)*[10,10,1]
a[:,2] = 3*a[:,0]+2*a[:,1]+np.random.rand(1,10)*2
s = "["
for row in a:
    s += "["
    for ele in row:
        s += f"{ele: .8f}"+"f32,"
    s = s[:-1]
    s += "],\n"
s = s[:-2]+"]"
np.savetxt("data.txt", a)
with open("data.fut","w") as f:
    f.write("let data="+s)
    s = "["
