from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(1000)

a = 0
b = 10
train_size = 100
test_size = 25
gaussian = 1

data_x = np.random.uniform(low=a, high=b, size=train_size+test_size)
data_y = np.sin(data_x)

train_x, train_y = data_x[:train_size], data_y[:train_size]
test_x, test_y = data_x[train_size:], data_y[train_size:]

lagrange_polynomial = lagrange(train_x, train_y)

training_error = np.mean(np.abs(lagrange_polynomial(train_x)-train_y))
testing_error = np.mean(np.abs(lagrange_polynomial(test_x)-test_y))
print("training err", training_error)
print("testing err", testing_error)


if gaussian:
    stddev = [0.01, 0.05, 0.1, 0.5, 1]
    train_err, test_err = [], []
    for eps in stddev:
        print("expt for eps =", eps)
        noise = np.random.normal(scale=eps, size=train_size+test_size)
        data_x = data_x + noise
        data_y = np.sin(data_x)

        train_x, train_y = data_x[:train_size], data_y[:train_size]
        test_x, test_y = data_x[train_size:], data_y[train_size:]

        lagrange_polynomial = lagrange(train_x, train_y)

        training_error = np.mean(np.abs(lagrange_polynomial(train_x)-train_y))
        train_err.append(training_error)
        testing_error = np.mean(np.abs(lagrange_polynomial(test_x)-test_y))
        test_err.append(testing_error)

    plt.plot(stddev, train_err, label='train')
    plt.plot(stddev, test_err, label='test')
    plt.legend()
    plt.xlabel('std. dev of the noise')
    plt.ylabel('error')
    plt.savefig("./plots/err vs stddev.pdf")