import copy

W1_snapshots = []
b1_snapshots = []
W2_snapshots = []
b2_snapshots = []

def save_neurons_snapshot(W1, b1, W2, b2):
    W1_snapshots.append(copy.deepcopy(W1))
    b1_snapshots.append(list(b1))
    W2_snapshots.append(copy.deepcopy(W2))
    b2_snapshots.append(list(b2))

def get_neurons_snapshots():
    return W1_snapshots, b1_snapshots, W2_snapshots, b2_snapshots
