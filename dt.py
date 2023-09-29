import numpy as np

cuts_for_root = []
count_leaf_nodes = 0
count_internal_nodes = 0

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, gain_ratio=None, label=None):

        # for internal
        self.j = feature
        self.c = threshold
        self.left_child = left
        self.right_child = right
        self.info_gain = info_gain
        self.gain_ratio = gain_ratio
        
        # for leaf
        self.label = label

class DecisionTree():
    def __init__(self):
        self.root_node = None
        self.count_leaf_nodes = 0
        self.count_internal_nodes = 0


    def MakeSubTree(self, dataset, depth=0):
        current_depth = depth
        X, Y = dataset[:, :-1], dataset[:, -1]
        candidate_splits, stop = self.get_candidate_splits(dataset)

        if stop:
            y = list(Y)
            leaf_label = max(set(y), key = y.count)
            print("creating leaf node at depth", current_depth)
            self.count_leaf_nodes += 1
            return Node(label=leaf_label) 
        
        else:
            best_split = self.get_best_split(dataset, candidate_splits)
            if depth == 0:
                # print("candidate cuts and info gain", cuts_for_root)
                pass
            left_subtree = self.MakeSubTree(best_split["left_tree_data"], current_depth+1)
            right_subtree = self.MakeSubTree(best_split["right_tree_data"], current_depth+1)
            self.count_internal_nodes += 1
            return Node(best_split["feature"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"], best_split["gain_ratio"]) 
        
        
    def get_candidate_splits(self, dataset):
        stop = False
        splits = []
        for j in range(dataset.shape[1]-1):
            c = []
            temp = dataset[dataset[:, j].argsort()]
            for i in range(temp.shape[0]-1):
                if temp[i, -1] != temp[i+1, -1]:
                    c.append(temp[i+1, j]) 
            # print("thresholds", len(c))
            splits.append(c)

        num_splits = sum(len(x) for x in splits)
        if num_splits == 0 or dataset.shape[0] == 1:
            stop = True
            return splits, stop
        return splits, stop
        
            
    def get_best_split(self, dataset, candidate_splits):
        best_split = {}
        max_gain_ratio = -1

        for j in range(len(candidate_splits)):
            for c in candidate_splits[j]:
                # print("j, c", j, c)
                left_data, right_data = self.split_data(dataset, j, c)
                # print("split data sizes", len(left_data), len(right_data))
                # print("finding max info gain for", j, c)
                if len(left_data)>0 and len(right_data)>0:    
                    # print("printing left, right datatset", left_data, right_data)
                    y, left_y, right_y = dataset[:, -1], left_data[:, -1], right_data[:, -1]
                    info_gain = self.get_info_gain(y, left_y, right_y)
                    split_entropy = self.get_split_entropy(dataset, left_data, right_data)
                    gain_ratio = info_gain / split_entropy
                    if gain_ratio == 0:
                        print("gain ratio zero!")
                    cuts_for_root.append([j, c, gain_ratio])
                    if gain_ratio > max_gain_ratio:
                        best_split["feature"] = j
                        best_split["threshold"] = c
                        best_split["left_tree_data"] = left_data
                        best_split["right_tree_data"] = right_data
                        best_split["info_gain"] = info_gain
                        best_split["gain_ratio"] = gain_ratio
                        max_gain_ratio = gain_ratio

        return best_split


    def split_data(self, dataset, j, c):
        left = np.array([data for data in dataset if data[j]>=c])
        right = np.array([data for data in dataset if data[j]<c])
        return left, right
        pass
    
    def get_split_entropy(self, node, left, right):
        p_left = len(left)/len(node)
        p_right = len(right)/len(node)
        return (-p_left * np.log2(p_left)) + (-p_right * np.log2(p_right))

    def get_info_gain(self, node, left, right):
        p_left = len(left)/len(node)
        p_right = len(right)/len(node)
        info_gain = self.entropy(node) - (p_left*self.entropy(left) + p_right*self.entropy(right))
        return info_gain
    
    def entropy(self, l):
        labels = np.unique(l)
        ent = 0
        for y in labels:
            p = len(l[l == y]) / len(l)
            ent += -p * np.log2(p)
        return ent
 
    
        


