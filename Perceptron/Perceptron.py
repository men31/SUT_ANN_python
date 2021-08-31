import numpy as np
import matplotlib.pyplot as plt

# Activation function.
def Hardlimit(x):
    return 1 if x>= 0 else 0

# Perceptron by vectorization method.
def PerceptronVectorize(X, y, theta, b, limit=20, CostFunc=False):
    iteration = 0
    J = []
    theta = np.vstack(([b], theta))
    VecHardlim = np.vectorize(Hardlimit)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    while iteration < limit:
        h = VecHardlim(X @ theta)
        e = y - h
        theta = theta + X.T @ e
        print('iter : ', iteration, ' theta : ', theta)
        if len(e.shape) > 1:
            cost = sum(sum(e**2))
        else:
            cost = sum(e**2)
        J.append(cost)
        if  cost == 0:
            print('Vectorize End Iteration : ', iteration)
            break
        iteration += 1
    if CostFunc:
        return theta[1:], theta[0], J
    return theta[1:], theta[0]

# Perceptron by Programming method.
def PerceptronNormal(X, y, W, b, limit=10, CostFunc=False):
    iteration = 0
    VecHardlim = np.vectorize(Hardlimit)
    J = []
    while iteration < limit:
        cost = np.array([])
        for i in range(len(X)):
            a = X[i] @ W.T + b
            # a = a.reshape(len(a), 1)
            # print(a.shape)
            h = VecHardlim(a)
            e = y[i] - h
            # print('cost : ', cost)
            # print(X[i].shape, '   ', W.shape)
            W = W + X[i].T * e
            b = b + e
            cost = np.append(cost, e)
            # print('iter : ', iteration, ' theta : ', W)
        J.append(sum(cost**2))
        if sum(cost**2) == 0:
            print('Normal End Iteration : ', iteration)
            break
        iteration += 1
    if CostFunc:
        return W, b, J
    return W, b, J     

def PerceptronCoursera(X, y, theta, b, alpha=0.8, limit=20, CostFunc=False):
    iteration = 0
    J = []
    theta = np.vstack(([b], theta))
    VecHardlim = np.vectorize(Hardlimit)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    while iteration < limit:
        h = VecHardlim(X @ theta)
        # print(h)
        cost = np.sum((h - y)**2) / (2*y.shape[0])
        theta = theta - alpha * sum((h - y) * X, 1) / (y.shape[0])
        # theta = theta - alpha * X.T @ (h - y)  / (y.shape[0])
        # print(sum(X.T @ (h - y)).shape)
        # print('theta : ', theta)
        if cost**2 == 0:
            print('Normal End Iteration : ', iteration)
            break
        J.append(cost)
        iteration += 1
    print('theta : ', theta)
    if CostFunc:
        return theta[1:], theta[0], J
    return theta[1:], theta[0]

if __name__ == '__main__':
    X = np.array([[-0.5, 2], [1, 2], [1.5, -0.5], [2, 1], [-2, -1], [-1, 1],
                    [-1, -1], [1, -1]])
    # X = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 1, 1, 1]).reshape(X.shape[0], 1)
    y = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1]])
    # y = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(X.shape[0], 1)
    # theta = np.array([[0, 0], [0, 0]])
    theta = np.zeros((2, 2))
    # theta = np.array([[-1.2], [-0.5]])
    b = np.array([1, 1])
    Lm = 100
    new_theta, new_b, JVec = PerceptronVectorize(X, y, theta, b, limit=Lm, CostFunc=True)
    print('VectorMethod : ', 'Theta : ',new_theta)
    print('B :', new_b)
    # W, new_b, JNor = PerceptronNormal(X, y, theta, b, limit=Lm, CostFunc=True)
    # print('NormalMethod : ', 'Theta : ', W)
    # print('B : ', new_b)
    # new_theta, new_b, JVec = PerceptronCoursera(X, y, theta, b, limit=Lm, CostFunc=True)
    # print('VectorMethod : ', 'Theta : ',new_theta)
    # print('B :', new_b)
    numVec = np.arange(len(JVec))
    # numNor = np.arange(len(JNor))
    plt.figure('Vecterization Cost Function')
    plt.plot(numVec, JVec, '.--')
    plt.title(f'Num of iteration {len(JVec)}')
    # plt.figure('Normal Cost Function')
    # plt.plot(numNor, JNor, '.--')
    # plt.title(f'Num of iteration {len(JNor)}')
    # plt.show()

    
    
