from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# fixed seed for reproducing results
np.random.seed(1000)

# get data
dataset = []
with open("./data/Dbig.txt", "r") as f:
    for l in f.readlines():
        data = l.split(" ")
        data[-1] = data[-1][:-1]
        data = [float(i) for i in data]
        dataset.append(data)
dataset = np.array(dataset)


test_error = []
num_nodes = []

# shuffle data
np.random.shuffle(dataset)

# create datasets
d8192, test = dataset[:8192, :], dataset[8192:, :]
d32 = d8192[:32, :]
d128 = d8192[:128, :]
d512 = d8192[:512, :]
d2048 = d8192[:2048, :]
test_x, test_y = test[:, :-1], test[:, -1]
for dataset in [d32, d128, d512, d2048, d8192]:
    # training
    train_x, train_y = dataset[:, :-1], dataset[:, -1]
    clf = DecisionTreeClassifier(max_depth=20, random_state=100)
    clf.fit(train_x, train_y)

    # testing and error
    predicted = clf.predict(test_x)
    error = 1 - accuracy_score(test_y, predicted)

    num_nodes.append(clf.tree_.node_count)
    test_error.append(error)
    print("test error", test_error)
    print("# nodes", num_nodes)

# plotting
plt.plot([32, 128, 512, 2048, 8192], test_error)
plt.xlabel("n")
plt.ylabel("test error")
plt.savefig("./plots/error vs training size-sklearn.pdf")