from dt import *
import numpy as np
import matplotlib.pyplot as plt

# get data
dataset = []
with open("./D1.txt", "r") as f:
    for l in f.readlines():
        data = l.split(" ")
        data[-1] = data[-1][:-1]
        data = [float(i) for i in data]
        dataset.append(data)


dataset = np.array(dataset)

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
            print("X_"+str(tree.j), ">=", tree.c, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            print_tree(tree.left_child, indent + indent)
            print("%sright:" % (indent), end="")
            print_tree(tree.right_child, indent + indent)



countFreq(dataset[:, -1], dataset.shape[0])
classifier = DecisionTree()
classifier.root_node = classifier.MakeSubTree(dataset)
# print_tree(classifier.root_node)

# plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, -1])
# plt.show()