import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random
import math
import sklearn

def cross_entropy_loss(X, c, W, b):
    # x: n * m, c: l * m, w: n * l, b: l
    n, m = np.shape(X)
    l = len(c)
    x = np.vstack((X, np.ones(m)))
    w = np.vstack((W, b))
    n += 1
    x_t = np.transpose(x)
    den = np.zeros(m) # will be the denomenator
    for j in range(l):
        den = den + np.exp(np.dot(x_t, w.T[j]))
    den = np.diag(den)
    den = np.linalg.inv(den)
    res = 0.0
    for k in range(l):
        numerator = np.exp(np.matmul(x_t, w.T[k]))
        ck_t = np.transpose(c[k])
        elem = np.matmul(ck_t, np.log(np.matmul(den, numerator)))
        res += elem
    return res/-m

# Gradient:
def gradient_w(X, c, W, b, p):
    # x: n * m, c: l * m, w: n * l, b: l
    n, m = np.shape(X)
    l = len(c)
    x = np.vstack((X, np.ones(m)))
    w = np.vstack((W, b))
    n += 1
    x_t = np.transpose(x)
    den = np.zeros(m) # will be the denomenator
    for j in range(l):
        den += np.exp(np.dot(x_t, w.T[j]))
    den = np.diag(den)
    den = np.linalg.inv(den)
    nom = np.exp(np.dot(x_t, w.T[p]))
    res = np.dot(den, nom) - c[p]
    res = (np.dot(x, res)) * (1 / m)
    return res

# Gradient Test:
def grad_test():
    mat_file = scipy.io.loadmat('HW1_Data/GMMData.mat')
    Yt, Ct = mat_file['Yt'], mat_file['Ct']
    Y = Yt[:, :50]
    C = Ct[:, :50]
    n = len(Y) # size of each datum
    m = len(Y[0]) # size of data
    l = len(C) # num of lables
    b = np.random.randn(l)
    def F(w):
        return cross_entropy_loss(Y, C, w, b)
    def g_F(w, p):
        return gradient_w(Y, C, w, b, p)
    x = np.random.randn(n , l) # x is w
    d = np.random.randn(n , l)
    d_ = np.ones(l)
    d_ = np.vstack((d, d_))
    d_ = d_.flatten('C') # matrix 2 vector of size [(n + 1) * l, 1]
    epsilon = 1
    F0 = F(x) # scalar
    G0 = np.zeros((l, n + 1))
    for p in range(l):
        g0 = g_F(x, p)
        G0[p] += g0
    G0 = G0.flatten('F') # gradient vector
    y0 = np.zeros(8)
    y1 = np.zeros(8)
    print("k\tF(x + e * d) - F(x)\t\tF(x + e * d) - F(x) - e * d.T * gradient")
    for k in range(8):
        epsk = epsilon * (0.5**k)
        Fk = F(x + epsk * d)
        F1 = F0 + epsk * (np.dot(G0, d_))
        y0[k] = abs(Fk - F0)
        y1[k] = abs(Fk - F1)
        print(k,"\t", y0[k],"\t", y1[k], "\n")
    plt.semilogy(np.arange(1, 9),y0)
    plt.semilogy(np.arange(1, 9),y1)
    plt.legend(("O(epsilon)","O(epsilon square)"))
    plt.title("Gradient Test Plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()
    pass
    grad_test()

# Stochastic Gradient Descent:
def stochastic_gradient_descent(X, Y, maxIter, batchSize, alpha):
    num_features, num_samples = X.shape
    num_classes, _ = Y.shape
    b = np.random.rand(num_classes)
    W = np.random.randn(num_features , num_classes)
    w = np.vstack((W, np.random.randn(num_classes)))
    loss_history = []
    for i in range(maxIter):
        # divide the indices into random mini-batches
        indices = np.random.permutation(num_samples)
        mini_batches = [indices[k:k+batchSize] for k in range(0, num_samples, batchSize)]
        # loop over mini-batches
        for mini_batch in mini_batches:
            # compute mini-batch gradient
            grad = np.zeros(num_features + 1)
            for j in mini_batch:
                x = X[:, :j + 1]
                y = Y[:, :j + 1] # C
                grad += gradient_w(x, y, W, b, 0)
            grad = grad / batchSize
            # apply a step
            for j_ in range(num_classes):
                if(j_ > 0):
                    w.T[j_] += w.T[j_ - 1] - alpha * grad
                    if(j_ < num_features):
                        W.T[j_] = w[:num_features, :].T[j_]
        loss_history.append(cross_entropy_loss(X, Y, W, b))
    return w, loss_history

# Minimizing Objective Function Using SGD with Momentum?
def sgd_momentum(w0, f, grad_f, lr=0.01, momentum=0.9, num_epochs=100):
    n1, n2 = np.shape(w0)
    w_ = np.copy(w0)
    w = np.vstack((w0, np.ones(n2)))
    m_ = np.zeros_like(n1 + 1)
    loss_history = []
    for i in range(num_epochs):
        grad = grad_f(w_)
        m_ = momentum * m_ + lr * grad # (1 - momentum) * grad
        m = np.array([m_ for k in range(n2)])
        w -= m.T # lr * m
        w_ = w[:n1,:]
        loss_history.append(f(w_))
    return w, loss_history

# Demonstarating The Minimization of The softmax Function Using SGD Variant:
maxIter = 500
batchSize = 10
alpha_values = [0.001, 0.01, 0.1]
fig_size = (16,48)
fig, axs = plt.subplots(1, 3, figsize=fig_size)
fig2, axs2 = plt.subplots(1, 3, figsize=fig_size)
fig.suptitle('Classification Results')
fig2.suptitle('Loss History (Per iteration on the data)')
for k, alpha in enumerate(alpha_values):
    # run SGD
    w, loss = stochastic_gradient_descent(Yt, Ct, maxIter, batchSize, alpha)
    # plot results
    ax = axs[1, k]
    Rt = np.apply_along_axis(lambda x: np.argmax(x), 1, np.dot(Yt.T, w))
    ax.scatter(Yt[0,:], Yt[1,:], c=Rt)
    ax.set_title('maxIter={}, batchSize={}, alpha={}'.format(maxIter, batchSize, alpha))
    ax2 = axs2[1, k]
    ax2.plot(loss)
    ax2.set_xlabel('Iter')
    ax2.set_ylabel('Loss')
    ax2.set_title('maxIter={}, batchSize={}, alpha={}'.format(maxIter, batchSize, alpha))
plt.show()

# Derivatives: Standard Neutral Network:
def derivative_part(W, x, b, v):
    # x: n * 1, w: k * n, b: k * 1, v: k * 1
    # x: n * m, W: k * n, b: k * m, v: k * m
    n, m = np.shape(x)
    k = len(W)
    res = np.zeros((k, m))
    for i in range(m):
        xi = x.T[i]
        bi = b.T[i]
        vi = v.T[i]
        teta = np.tanh(np.dot(W, xi) + bi)
        teta_derivative = (np.ones(k) - (teta)**2) # teta'(Wx + b)
        # teta'(Wx + b) (.) v
        J_T_v = np.multiply(vi, teta_derivative.T)
        res.T[i] = J_T_v
    return res

def c_derivative_wrt_b(W, x, b , v):
    return derivative_part(W, x, b , v)

def c_derivative_wrt_W(W, x, b , v):
    J_T_v = derivative_part(W, x, b , v)
    J_T_v = np.dot(J_T_v, x.T)
    return J_T_v

def c_derivative_wrt_x(W, x, b , v):
    J_T_v = derivative_part(W, x, b , v)
    J_T_v = np.dot(W.T, J_T_v)
    return J_T_v

# The Jacobian Transposed Test: Standard Neutral Network:
def Jacobian_Transposed_test_standart_network():
    mat_file = scipy.io.loadmat('HW1_Data/GMMData.mat')
    Yt, Ct = mat_file['Yt'], mat_file['Ct']
    Y = Yt[:, :50]
    n, m = np.shape(Y) # size of each datum, size of data
    l = np.random.randint(m / 2)
    q = np.random.randint(m)
    q0 = np.random.randint(n)
    print("n: ", n, " m: ", m, " l: ", l)
    b = np.random.randn(l, m)
    v = np. random.randn(l, m)
    def F(w):
        g = 0 # inner product
        for p in range(l):
            teta = np.tanh(np.dot(w[p], Y) + b[p])
            g += np.dot(teta, v[p].T)
        return g
    def g_F(w):
        return c_derivative_wrt_W(w, Y, b , v)
    x = np.random.randn(l, n) # x is w
    d = np.random.randn(l, n)
    d_ = d.flatten('F')
    epsilon = 1
    F0 = F(x)
    G0 = g_F(x) # l * n
    G0 = G0.flatten('F')
    y0 = np.zeros(8)
    y1 = np.zeros(8)
    print("k\tF(x + e * d) - F(x)\t\tF(x + e * d) - F(x) - e * d.T * gradient")
    for k in range(8):
        epsk = epsilon * (0.5**k)
        Fk = F(x + epsk * d) # np.zeros(l)
        # for p in range(l):
        # Fk[p] = F(x + epsk * d, p)
        F1 = F0 + epsk * np.dot(G0, d_)
        # for i in range(n):
        # F1 += np.dot(d.T[i], G0)
        # F1 = F0 + epsk * F1
        y0[k] = abs(Fk - F0)
        y1[k] = abs(Fk - F1)
        print(k,"\t", y0[k],"\t", y1[k], "\n")
    plt.semilogy(np.arange(1, 9),y0)
    plt.semilogy(np.arange(1, 9),y1)
    plt.legend(("O(epsilon)","O(epsilon square)"))
    plt.title("Jacobian Transposed Test Plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()
    pass

# The Standard Neural Network:
def forward(W, b, X, C, activation=np.tanh):
    len_W = len(W) - 1
    a = []
    z = []
    a_ = X
    a.append(a_)
    for layer in range(len_W):
        na, ma = np.shape(a_)
        kb = len(b[layer])
        z_ = np.zeros((ma, kb))
        for m in range(ma):
            z_[m] = np.dot(W[layer].T, a_.T[m]) + b[layer]
        z.append(z_.T)
        a_ = activation(z_.T)
        a.append(a_)
    y_hat = cross_entropy_loss(a_, C, W[len_W], b[len_W])
    return a, y_hat

def backward(W, b, a, C):
    l = len(W)
    gradients_w = []
    gradients_b = []
    # calculation of last layer gradient
    size_Wl, size_Wl_T = np.shape(W[l - 1])
    size_al, size_al_T = np.shape(a[l - 1])
    last_grad_w = np.zeros((size_Wl_T, size_Wl))
    last_grad_b = np.zeros(size_Wl_T)
    for p in range(size_Wl_T): #(size_Wl_T):
        grad = gradient_w(a[l - 1], C, W[l - 1], b[l - 1], p)
        len_grad = len(grad)
        last_grad_w[p] += grad[: len_grad - 1]
        last_grad_b[p] += grad[len_grad - 1]
    gradients_w.append(last_grad_w)
    gradients_b.append(last_grad_b)
    # calculation of v1
    v = np.zeros((size_Wl, size_al_T))
    vi = np.random.randn(size_Wl)
    for i in range(size_al_T):
        v.T[i] += vi
    for i_ in range(l - 1):
        i = l - i_ - 2 # begin with the last layer
        if i < 0:
            break
        wi = W[i].T
        xi = a[i]
        bi = np.zeros((len(b[i]), len(xi[0])))
        for j in range(len(xi[0])):
            bi.T[j] += b[i]
        grad_w = c_derivative_wrt_W(wi, xi, bi, v) # grad_w[i] ~ w[l - i - 1]
        grad_b = c_derivative_wrt_b(wi, xi, bi, v)
        gradients_w.append(grad_w) # len( self.gradients_w) = len(W)
        gradients_b.append(grad_b)
        if i != 0:
         v = c_derivative_wrt_x(wi, xi, bi, v)
    return gradients_w, gradients_b

def scalar_list_multiplication(e, d):
    ans = []
    for i in range(len(d)):
        ans.append(e * d[i])
    return ans

def update(W, b, gradients_w, gradients_b, learning_rate):
    l = len(W)
    W[0] += learning_rate * gradients_w[l - 1].T
    b[0] += learning_rate * gradients_b[l - 1].T[0]
    for i_ in range(l - 1):
        i = l - i_ - 1
        if i < 0:
            break
        grad_wi_ = gradients_w[i_ + 1]
        len_gwi_ = len(grad_wi_) - 1
        W[i] += learning_rate * gradients_w[i_].T
        b[i] += learning_rate * gradients_b[i_].T[0]
    return W, b

def train(W, b, X, C, epochs, batch_size, learning_rate):
    losses = []
    W_, b_ = W, b
    for epoch in range(epochs):
        m = len(X[0])
        indices = np.random.permutation(m)
        mini_batches = [indices[k:k+batch_size] for k in range(0, m, batch_size)]
        loss = 0
        # loop over mini-batches
        for i, mini_batch in enumerate(mini_batches):
            Xb = X[:, mini_batch]
            Cb = C[:, mini_batch]
            a, l_ = forward(W_, b_, Xb, Cb)
            loss += l_ / (i + 1)
            gradients_w, gradients_b = backward(W_, b_, a, Cb)
            W_, b_ = update(W_, b_, gradients_w, gradients_b, learning_rate)
        losses.append(loss)
    return losses, a, W_, b_

# Derivatives: Residual Neutral Network:
def r_derivative_wrt_b(W1, W2, x, b , v):
    # x: n * 1, W1: n * n, W2: n * n, b: n * 1, v: n * 1
    # x: n * k, W1: n * n, W2: n * n, b: n * k, v: n * k
    n, m = np.shape(x)
    k = len(b)
    res = []
    for i in range(n):
        xi = x.T[i]
        bi = b.T[i]
        vi = v.T[i]
        W2_Ti = W2[i]
        teta = np.tanh(np.dot(W1, xi) + bi)
        teta_derivative = (np.ones(k) - (teta)**2)
        J_T = np.dot(teta_derivative, W2_Ti)
        J_T_v = np.dot(J_T, vi)
        res.append(J_T_v)
    return res

def r_derivative_wrt_W1(W1, W2, x, b , v):
    Jb = r_derivative_wrt_b(W1, W2, x, b , v)
    m = len(b[0])
    n = len(v)
    res = []
    for i in range(n):
        xi = x.T[i]
        Ji = Jb[i]
        J_T_v = np.dot(Ji, xi)
        res.append(J_T_v)
    return res

def r_derivative_wrt_W2(W1, W2, x, b , v):
    n, m = np.shape(x)
    k = len(b)
    res = np.zeros(np.shape(b.T))
    for i in range(n):
        xi = x.T[i]
        bi = b.T[i]
        teta = np.tanh(np.dot(W1, xi) + bi)
        res[i] = teta
    res = np.dot(v, res)
    return res

def r_derivative_wrt_x(W1, W2, x, b , v):
    n, m = np.shape(x)
    k = len(b)
    res = []
    d_w2 = np.zeros((n, k))
    teta = np.tanh(np.dot(W1, x) + b)
    teta_derivative = (np.ones(np.shape(b)) - (teta)**2)
    for i in range(n):
        ti = teta_derivative.T[i]
        w2i = W2.T[i]
        diag = np.diag(ti)
        d_w2i = np.dot(diag, w2i)
        d_w2[i] = d_w2i
    w1_d_w2 = np.dot(W1.T, d_w2)
    I = np.identity(len(w1_d_w2))
    J_T = np.add(I, w1_d_w2)
    J_T_v = np.dot(J_T, v)
    res.append(J_T_v)
    return res

# The Residual Block:
def pass_forward(W1, W2, b, X, activation):
    x1 = X
    _, m1 = np.shape(x1)
    k = len(b)
    z1 = np.zeros((m1, k))
    for m in range(m1):
        z1[m1] = np.dot(W1.T, x1.T[m1]) + b
    x2 = activation(z)
    _, m2 = np.shape(x2)
    z2 = np.zeros((m2, k))
    for m in range(m2):
        z2[m2] = np.dot(W2.T, x2.T[m2]) + b
    ans = activation(x1 + z2)
    return ans

def pass_backward(W1, W2, b, a, v):
    for j in range(len(a[0])):
        b.T[j] += b
    grad_W1 = r_derivative_wrt_W1(W1, W2, a, b, v)
    grad_W2 = r_derivative_wrt_W2(W1, W2, a, b, v)
    grad_b = r_derivative_wrt_b(W1, W2, a, b, v)
    grad_x = r_derivative_wrt_x(W1, W2, a, b, v)

# The Residual Neural Network:
def forward(W1, W2, b, X, C, activation=np.tanh):
    net_len = len(W1) - 1
    a = [X]
    for layer in range(net_len):
        a.append(pass_forward(W1[layer], W2[layer], b[layer], a[layer], activation))
    y_hat = cross_entropy_loss(a[net_len], C, W1[net_len], b[net_len])
    return a, y_hat

def backward(W1, W2, b, a, C):
    l = len(W1)
    gradients_w1 = []
    gradients_w2 = []
    gradients_b = []
    # calculation of last layer gradient
    size_Wl, size_Wl_T = np.shape(W1[l - 1])
    _, size_al_T = np.shape(a[l - 1])
    last_grad_w = np.zeros((size_Wl_T, size_Wl))
    last_grad_b = np.zeros(size_Wl_T)
    for p in range(size_Wl_T): #(size_Wl_T):
        grad = gradient_w(a[l - 1], C, W1[l - 1], b[l - 1], p)
        len_grad = len(grad)
        last_grad_w[p] += grad[: len_grad - 1]
        last_grad_b[p] += grad[len_grad - 1]
    gradients_w1.append(last_grad_w)
    gradients_b.append(last_grad_b)
    # calculation of v1
    v = np.zeros((size_Wl, size_al_T))
    vi = np.random.randn(size_Wl)
    for i in range(size_al_T):
        v.T[i] += vi
    for i_ in range(l - 1):
        i = l - i_ - 2 # begin with the last layer
        if i < 0:
            break
        grad_W1, grad_W2, grad_b, v = pass_backward(W1[i], W2[i], b[i], a[i], v)
        gradients_w1.append(grad_W1)
        gradients_w2.append(grad_W2)
        gradients_b.append(grad_b)
    return gradients_w1, gradients_w1, gradients_b

def scalar_list_multiplication(e, d):
    ans = []
    for i in range(len(d)):
        ans.append(e * d[i])
    return ans

def update(W1, W2, b, gradients_w1, gradients_w2, gradients_b, learning_rate):
    l = len(W1)
    W1[0] += learning_rate * gradients_w1[l - 1].T
    W2[0] += learning_rate * gradients_w2[l - 2].T
    b[0] += learning_rate * gradients_b[l - 1].T[0]
    for i_ in range(l - 1):
        i = l - i_ - 1
        if i < 0:
            break
        W1[i] += learning_rate * gradients_w1[i_].T
        b[i] += learning_rate * gradients_b[i_].T[0]
        if(i > 0):
            W2[i-1] += learning_rate * gradients_w1[i_].T
    return W1, W2, b

def train(W1, W2, b, X, C, epochs, batch_size, learning_rate):
    losses = []
    for epoch in range(epochs):
        m = len(X[0])
        indices = np.random.permutation(m)
        mini_batches = [indices[k:k+batch_size] for k in range(0, m, batch_size)]
        loss = 0
        # loop over mini-batches
        for i, mini_batch in enumerate(mini_batches):
            Xb = X[:, mini_batch]
            Cb = C[:, mini_batch]
            a, l_ = forward(W1, W2, b, Xb, Cb)
            loss += l_ / (i + 1)
            gradients_w1, gradients_w2, gradients_b = backward(W1, W2, b, a, Cb)
            W1, W2, b = update(W1, W2, b, gradients_w1, gradients_w2, gradients_b, learning_rate)
        losses.append(loss)
    return losses, a, W1, W2, b