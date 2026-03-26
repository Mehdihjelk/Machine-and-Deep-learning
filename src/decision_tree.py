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

