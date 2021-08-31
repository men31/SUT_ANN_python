from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

def Hardlimit(x):
    return 1 if x>= 0 else 0
VecHardlim = np.vectorize(Hardlimit)



X = np.array([[-0.5, 2], [1, 2], [1.5, -0.5], [2, 1], [-2, -1], [-1, 1],
                    [-1, -1], [1, -1]])
y = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1]])
# y = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 1, 1, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 1, 1, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 1, 1]]).T
theta = np.array([[1, 0], [-4, 3.5], [-4.5, -5]])
# X = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])
# y = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(X.shape[0], 1)
# theta = np.array([[-1], [-4], [4], [4]])
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 1, 1, 1]).reshape(X.shape[0], 1)
# theta = np.array([[-1], [1.8], [1.5]])
# Separate_dot = np.linspace(-1, 1, 50).reshape(50, 1)
# Xtest = np.hstack((Separate_dot, Separate_dot))
# Xtest = np.hstack((np.ones((50, 1)), Xtest))
X_new = np.hstack((np.ones((X.shape[0], 1)), X))
result = VecHardlim(X_new @ theta)
print(result)
# test_data = Xtest @ theta
plt.plot(X[:, 0], X[:, 1], 'd')
# plt.plot(np.arange(len(test_data)), test_data[:,0], '.--')
plt.show()

