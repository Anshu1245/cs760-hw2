from dt import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1000)

def countFreq(arr, n):
    visited = [False for i in range(n)]

    for i in range(n):
        if (visited[i] == True):
            continue
 
        count = 1
        for j in range(i + 1, n, 1):
            if (arr[i] == arr[j]):
                visited[j] = True
                count += 1
         
        print(arr[i], count)

def print_tree(classifier, indent=" "):        
        tree = classifier

        if tree.label is not None:
            print(tree.label)

        else:
            print("X_"+str(tree.j), ">=", tree.c, "?", tree.info_gain, tree.gain_ratio)
            print("%sleft:" % (indent), end="")
            print_tree(tree.left_child, indent + indent)
            print("%sright:" % (indent), end="")
            print_tree(tree.right_child, indent + indent)

def predict(x, dt):
    if dt.label!=None: return dt.label
    value = x[dt.j]
    if value>=dt.c:
        return predict(x, dt.left_child)
    else:
        return predict(x, dt.right_child)


def get_test_error(test_X, test_Y, dt):
    error = 0
    for i in range(len(test_X)):
        y = predict(test_X[i], dt)
        error += (y!=test_Y[i])
    return error/len(test_X)




# get data
dataset = []
dbig = 1
with open("./data/Dbig.txt", "r") as f:
    for l in f.readlines():
        data = l.split(" ")
        data[-1] = data[-1][:-1]
        data = [float(i) for i in data]
        dataset.append(data)


dataset = np.array(dataset)
# print(dataset)

countFreq(dataset[:, -1], dataset.shape[0])

if dbig:
    test_error = []
    num_nodes = []
    np.random.shuffle(dataset)
    d8192, test = dataset[:8192, :], dataset[8192:, :]
    d32 = d8192[:32, :]
    d128 = d8192[:128, :]
    d512 = d8192[:512, :]
    d2048 = d8192[:2048, :]

    for dataset in [d32, d128, d512, d2048, d8192]:
        classifier = DecisionTree()
        classifier.root_node = classifier.MakeSubTree(dataset)
        num_nodes.append(classifier.count_internal_nodes + classifier.count_leaf_nodes)
        test_error.append(get_test_error(test[:, :-1], test[:, -1], classifier.root_node))
        print("test error", test_error)
        print("# nodes", num_nodes)

    plt.plot([32, 128, 512, 2048, 8192], test_error)
    plt.xlabel("n")
    plt.ylabel("test error")
    plt.savefig("./plots/error vs training size.pdf")


else:
    classifier = DecisionTree()
    classifier.root_node = classifier.MakeSubTree(dataset)
    print_tree(classifier.root_node)



# plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, -1], label=dataset[:, -1])
# # plt.axhline(y=0.201829, color='r')
# plt.xlabel("X0")
# plt.ylabel("X1")
# plt.show()