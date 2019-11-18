
import numpy as np

angle = 45

trans = [[ np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))],
         [-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]]

point = [4,4]

trans_point = np.asarray(trans).dot(np.asarray(point))
print(trans)
print(point)
print(trans_point)