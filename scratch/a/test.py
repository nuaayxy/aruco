import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("error.txt",delim_whitespace=True)
data1 = pd.read_csv("hmd.txt",delim_whitespace=True)
data2 = pd.read_csv("hmd1.txt",delim_whitespace=True)
varjo = pd.read_csv("varjo.txt",delim_whitespace=True)

error = data.to_numpy()
error1 = data1.to_numpy()
offset = data2.to_numpy()
varjo = varjo.to_numpy() * 10

print(np.shape(error))
mag = np.linalg.norm(error,axis = 1)
mag = mag[mag<40]





print(np.mean(mag))
fig = plt.plot(mag)
plt.title("marker pos error iphone - varjo")
plt.ylabel("mm")
plt.draw()
plt.show()

varjo = varjo - varjo[0]
varjo = np.linalg.norm(varjo,axis = 1)
varjo  = varjo[varjo < 30]
fig = plt.plot(varjo)
plt.title("varjo marker position ")
plt.ylabel("mm")
plt.show()


imgAnchor = pd.read_csv("image.txt",delim_whitespace=True)
iphone = pd.read_csv("iphone.txt",delim_whitespace=True)
imgAnchor = imgAnchor.to_numpy() * 10
iphone = iphone.to_numpy() * 10


imgAnchor = imgAnchor - imgAnchor[0]
imgAnchor = np.linalg.norm(imgAnchor,axis = 1)
fig2 = plt.plot(imgAnchor)
# displaying the title
plt.title("image anchor position")
plt.show()



error_mag = np.linalg.norm(error1,axis = 1)
# error_mag = error_mag[error_mag < 200]
# error_mag = error_mag[error_mag < 100]
fig1 = plt.plot(error_mag)
plt.title("iphone-hmd distance")
plt.ylabel("mm")
plt.draw()
plt.show()


error_mag = np.linalg.norm(offset,axis = 1)
# error_mag = error_mag[error_mag < 200]
# error_mag = error_mag[error_mag < 100]
fig1 = plt.plot(error_mag)
plt.title("iphone-hmd offset distance")
plt.ylabel("mm")
plt.draw()
plt.show()



plt.title("iphone position")
iphone = iphone- iphone[0]
# iphone = np.linalg.norm(iphone, axis = 1)
plt.plot(iphone[:,1])
plt.show()

headset = pd.read_csv("headset.txt",delim_whitespace=True)
headset = headset.to_numpy() * 10
headset = headset - headset[0]
# headset = np.linalg.norm(headset, axis = 1)
plt.title("headset")
plt.plot(headset[:,1])
plt.draw()
