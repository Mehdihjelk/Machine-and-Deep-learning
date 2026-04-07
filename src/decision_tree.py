import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

    def isleaf(self):
        return self.value is not None

class regressiontree:
    def __init__(self, min_sample_split=2, max_depth=100):
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.root=None

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y, n_features)

        if best_feat is not None:
            left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
            
            left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
            right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
            
            return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

        return Node(value=np.mean(y))

    def _best_split(self, X, y, n_features):
        best_reduction = -1
        split_idx, split_threshold = None, None

        # On boucle sur chaque colonne
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            # On teste chaque valeur unique de cette colonne comme seuil potentiel
            thresholds = np.unique(X_column)
            
            for thr in thresholds:
                left_idxs, right_idxs = self._split(X_column, thr)
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                reduction = self._variance_reduction(y, y[left_idxs], y[right_idxs])
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _variance_reduction(self, y, y_left, y_right):
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        reduction = np.var(y) - (weight_left * np.var(y_left) + weight_right * np.var(y_right))
        return reduction

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])