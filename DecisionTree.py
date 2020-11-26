# Reference:
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py
import numpy as np
from collections import Counter


def entropy(y):
    hist = np.bincount(y)  # how many times each class appeared
    probs = hist / len(y)  # the probability of each class appeared
    return -np.sum([p * np.log2(p) for p in probs if p > 0])


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        """
        :param min_samples_split: minimum samples that required to further split
        :param max_depth: max depth of the tree
        :param n_feats: number of features
        """
        self.min_sample_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None  # start point of the tree

    def _grow_tree(self, x, y, depth=0):
        """
        :param x:
        :param y:
        :param depth: keep track of depth
        :return:
        """
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        # stop criteria
        if (depth >= self.max_depth
                or n_labels == 1  # only has one class at this node
                or n_samples < self.min_sample_split):  # not enough node to split
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # generate the best feature indexes and best threshold
        best_feat, best_thresh = self._best_criteria(x, y, feat_idxs)

        # use best feature indexes and best threshold to split left and right indexes
        left_idxs, right_idxs = self._split(x[:, best_feat], best_thresh)

        # split left tree and right tree by recursion
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, x, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            x_column = x[:, feat_idx]
            thresholds = np.unique(x_column)  # pick the unique thresholds

            # go through all possible thresholds once
            for threshold in thresholds:
                gain = self._information_gain(y, x_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, x_column, split_thresh):
        # compute parent Entropy
        parent_E = entropy(y)
        # generate split left, right
        left_idxs, right_idxs = self._split(x_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # weighted average child E
        n = len(y)
        n_left = len(left_idxs)
        n_right = len(right_idxs)
        # compute left entropy, right entropy
        E_l, E_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        # weighted average child E
        child_E = (n_left / n) * E_l + (n_right / n) * E_r
        # return information gain
        ig = parent_E - child_E
        return ig

    @staticmethod
    def _split(x_column, split_threh):
        left_idxs = np.argwhere(x_column <= split_threh).flatten()  # split left data indexes
        right_idxs = np.argwhere(x_column > split_threh).flatten()  # split right data indexes
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        # go left or go right
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def train(self, x, y):
        # safety check
        if not self.n_feats:  # if number of features is not none
            self.n_feats = x.shape[1]  # number of feature = actually number of feature
        else:
            self.n_feats = min(self.n_feats, x.shape[1])  # keep number of feature less than actually number of feature

        # grow tree
        self.root = self._grow_tree(x, y)

    @staticmethod
    def _most_common_label(y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


