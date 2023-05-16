import numpy as np

def node_split(d, t, i, v, return_data=False):
    if isinstance(v, (int, float, np.int32, np.int64, np.float32, np.float64)):
        m = d[:,i] < v
    elif type(v) is str:
        m = d[:,i] == v
    else:
        raise TypeError(f"Unknown type {type(v)} for split value {v}")
        
    if not return_data:
        return t[m], t[~m]
    else:
        return d[m], t[m], d[~m], t[~m]
    
def calc_gini(d, t, i, v):
    left, right = node_split(d, t, i, v)
    
    gini_left = 1 - ((np.unique(left, return_counts=True)[1] / len(left)) ** 2).sum()
    gini_right = 1 - ((np.unique(right, return_counts=True)[1] / len(right)) ** 2).sum()

    
    return (gini_left * len(left) + gini_right * len(right)) / len(t)

def best_split(d, t):
    i_best = 0
    v_best = 0
    gini_best = 1
    for i in range(d.shape[1]):
        values = np.unique(d[:,i])
        for v in values:
            gini = calc_gini(d, t, i, v)
            if gini < gini_best:
                i_best = i
                v_best = v
                gini_best = gini
    return i_best, v_best

class DecisionNode:
    def __init__(self, i, v):
        self.i = i
        self.v = v
        self.is_leaf = False
            
    def build_children(self, d, t):
        if len(np.unique(t)) == 1:
            self.is_leaf = True
            self.t = t[0]
            return
        
        d_l, t_l, d_r, t_r = node_split(d, t, self.i, self.v, return_data=True)
        
        self.left = DecisionNode(*best_split(d_l, t_l))
        self.left.build_children(d_l, t_l)
        
        self.right = DecisionNode(*best_split(d_r, t_r))
        self.right.build_children(d_r, t_r)
        
        
    def chose_child(self, d):
        if type(self.v) is str:
            return self.left if d[self.i] == self.v else self.right
        else:
            return self.left if d[self.i] < self.v else self.right
        
        
    def get_t(self, d):
        if self.is_leaf:
            return self.t
        
        child = self.chose_child(d)
        return child.get_t(d)

class DecisionTree:
    def fit(self, data, target):
        i, v = best_split(data, target)
        self.root = DecisionNode(i, v)
        self.root.build_children(data, target)
        
    def predict(self, data):
        return list(map(lambda x: self.root.get_t(x), data))