import pandas as pd
import numpy as np
import sys
from collections import Counter

# Node structure
class TreeNode:
    def __init__(self, leaf=False, class_label=None):
        self.leaf = leaf
        self.class_label = class_label          # leaf only
        self.feature_idx = None                 # index of column
        self.feature_name = None
        self.is_numeric = False
        self.threshold = None                   # for numeric splits
        self.categorical_values = None          # list of branch values (for display)
        self.children = {}                      # key = value (str) -> TreeNode

    def set_leaf(self, class_label):
        self.leaf = True
        self.class_label = class_label

    def set_numeric_node(self, feature_idx, feature_name, threshold, left_child, right_child):
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.is_numeric = True
        self.threshold = threshold
        self.children = {
            f"<= {threshold:.2f}": left_child,
            f"> {threshold:.2f}": right_child
        }

    def set_categorical_node(self, feature_idx, feature_name, children_dict):
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.is_numeric = False
        self.children = children_dict
        self.categorical_values = list(children_dict.keys())



# Entropy & Information Gain
def entropy(labels):
    """Calculate the entropy of a set of labels (base 2)."""
    if len(labels) == 0:
        return 0.0
    counts = Counter(labels)
    probs = np.array([c / len(labels) for c in counts.values()])
    return -np.sum(probs * np.log2(probs + 1e-12))   # tiny epsilon to avoid log(0)

def information_gain(data, target, feature, feature_type):
    """
    Compute information gain for a feature.
    feature_type: 'categorical' or 'numeric'.
    Returns: (gain, best_threshold_or_None, children_data_dict)
      - for numeric: children_data_dict = {'<= th': data_left, '> th': data_right}
      - for categorical: children_data_dict = {value: subset}
    """
    parent_entropy = entropy(data[target])
    gain = 0.0
    best_threshold = None
    children = {}

    if feature_type == 'categorical':
        # Multi‑way split on each distinct value
        values = data[feature].unique()
        weighted_entropy = 0.0
        for v in values:
            subset = data[data[feature] == v]
            weight = len(subset) / len(data)
            weighted_entropy += weight * entropy(subset[target])
            children[v] = subset
        gain = parent_entropy - weighted_entropy
        return gain, None, children

    else:  # numeric
        # Binary split using thresholds = midpoints between sorted unique values
        sorted_vals = np.sort(data[feature].unique())
        if len(sorted_vals) <= 1:          # all values equal
            return 0.0, None, {}
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        best_gain = -1
        best_split = None
        for th in thresholds:
            left = data[data[feature] <= th]
            right = data[data[feature] > th]
            # Avoid empty child (should not happen with midpoints unless duplicate extremities)
            if len(left) == 0 or len(right) == 0:
                continue
            left_ent = entropy(left[target])
            right_ent = entropy(right[target])
            weighted = (len(left)/len(data))*left_ent + (len(right)/len(data))*right_ent
            current_gain = parent_entropy - weighted
            if current_gain > best_gain:
                best_gain = current_gain
                best_threshold = th
                best_split = (left, right)
        if best_split is None:
            return 0.0, None, {}
        gain = best_gain
        children = {
            f"<= {best_threshold:.2f}": best_split[0],
            f"> {best_threshold:.2f}": best_split[1]
        }
        return gain, best_threshold, children
    
    

# Tree building
def build_tree(data, target, feature_names, feature_types, available_features=None):
    """
    Recursive ID3/C4.5 style tree builder.
    data: DataFrame
    target: name of target column
    feature_names: list of all feature column names
    feature_types: dict {name: 'categorical' or 'numeric'}
    available_features: list of feature indices currently allowed (categorical removal).
    Returns: TreeNode
    """
    if available_features is None:
        available_features = list(range(len(feature_names)))

    # 1. All examples have same label -> leaf
    if len(data[target].unique()) == 1:
        leaf = TreeNode(leaf=True, class_label=data[target].iloc[0])
        return leaf

    # 2. No features left or all features have zero gain -> majority leaf
    if len(available_features) == 0:
        majority = data[target].mode().iloc[0]
        leaf = TreeNode(leaf=True, class_label=majority)
        return leaf

    # 3. Find best feature by information gain
    best_gain = -1
    best_feature_idx = -1
    best_threshold = None
    best_children_data = {}
    best_is_numeric = False

    for idx in available_features:
        fname = feature_names[idx]
        ftype = feature_types[fname]
        gain, thresh, children_data = information_gain(data, target, fname, ftype)
        if gain > best_gain:
            best_gain = gain
            best_feature_idx = idx
            best_threshold = thresh
            best_children_data = children_data
            best_is_numeric = (ftype == 'numeric')

    # 4. If no useful split, return majority leaf
    if best_gain <= 0 or best_feature_idx == -1:
        majority = data[target].mode().iloc[0]
        leaf = TreeNode(leaf=True, class_label=majority)
        return leaf

    best_feature_name = feature_names[best_feature_idx]

    # 5. Create node and recurse
    if best_is_numeric:
        node = TreeNode()
        # recurse left and right (feature can be reused, so available_features unchanged)
        left_child = build_tree(best_children_data[f"<= {best_threshold:.2f}"],
                                target, feature_names, feature_types,
                                available_features)
        right_child = build_tree(best_children_data[f"> {best_threshold:.2f}"],
                                 target, feature_names, feature_types,
                                 available_features)
        node.set_numeric_node(best_feature_idx, best_feature_name,
                              best_threshold, left_child, right_child)
    else:
        node = TreeNode()
        # remove this categorical feature from future consideration
        new_available = [i for i in available_features if i != best_feature_idx]
        children_nodes = {}
        for value, subset in best_children_data.items():
            child = build_tree(subset, target, feature_names, feature_types,
                               new_available)
            children_nodes[value] = child
        node.set_categorical_node(best_feature_idx, best_feature_name,
                                  children_nodes)
    return node



# Text representation
def print_tree(node, feature_names, target_name, depth=0, prefix=""):
    """Print a text‑based tree with indentation."""
    if node.leaf:
        print(f"{prefix}[Class: {node.class_label}]")
        return

    if node.is_numeric:
        print(f"{prefix}[{node.feature_name} <= {node.threshold:.2f}]")
        for branch, child in node.children.items():
            new_prefix = prefix + "  "
            print(f"{new_prefix}{branch}:")
            print_tree(child, feature_names, target_name, depth+1, new_prefix + "  ")
    else:
        print(f"{prefix}[{node.feature_name}]")
        for value, child in node.children.items():
            new_prefix = prefix + "  "
            print(f"{new_prefix}{value}:")
            print_tree(child, feature_names, target_name, depth+1, new_prefix + "  ")



# Graphical visualization using networkx & matplotlib
def plot_tree(node, feature_names, target_name, title="Decision Tree"):
    """Plot the tree using networkx and matplotlib with a custom tree layout."""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    pos = {}   # positions: node_id -> (x, y)

    # Count leaves to determine horizontal spacing
    def count_leaves(n):
        if n.leaf:
            return 1
        return sum(count_leaves(child) for child in n.children.values())

    total_leaves = count_leaves(node)

    # Inorder traversal to assign x coordinates to leaves,
    # then place internal nodes as average of children's x positions.
    leaf_counter = 0

    def assign_positions(n, depth=0):
        nonlocal leaf_counter
        if n.leaf:
            x = leaf_counter
            leaf_counter += 1
            y = -depth
            pos[id(n)] = (x, y)
            return x
        else:
            # process children in a fixed order
            # for numeric: left branch first (<=), then right (>)
            # for categorical: sorted by value to keep consistent
            child_x = []
            if n.is_numeric:
                # keys: '<= th', '> th' – order is <= then >
                for branch_key in [f"<= {n.threshold:.2f}", f"> {n.threshold:.2f}"]:
                    child = n.children[branch_key]
                    child_x.append(assign_positions(child, depth + 1))
            else:
                # categorical, sort branch values
                for value in sorted(n.children.keys()):
                    child_x.append(assign_positions(n.children[value], depth + 1))
            x = sum(child_x) / len(child_x)
            y = -depth
            pos[id(n)] = (x, y)
            return x

    assign_positions(node)

    # Add nodes and edges
    def add_to_graph(n):
        node_id = id(n)
        if n.leaf:
            label = f"Class: {n.class_label}"
            G.add_node(node_id, label=label, shape='ellipse', color='lightgreen')
        else:
            if n.is_numeric:
                label = f"{n.feature_name}\n≤ {n.threshold:.2f}?"
            else:
                label = n.feature_name
            G.add_node(node_id, label=label, shape='box', color='lightblue')
            for branch, child in n.children.items():
                child_id = id(child)
                G.add_edge(node_id, child_id, label=str(branch))
                add_to_graph(child)

    add_to_graph(node)

    # Draw
    plt.figure(figsize=(max(8, total_leaves * 1.2), max(5, np.log2(total_leaves+1)*2)))
    ax = plt.gca()
    ax.axis('off')

    # Use custom layout
    # Draw nodes with different shapes
    node_shapes = {}
    node_colors = {}
    for n_id, attrs in G.nodes(data=True):
        node_shapes[n_id] = attrs.get('shape', 'box')
        node_colors[n_id] = attrs.get('color', 'lightblue')

    # Draw edges with labels
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>',
                           edge_color='gray')
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                                 ax=ax, label_pos=0.5)

    # Draw rectangle nodes (all use the same shape here, we just draw rounded boxes via plain boxes)
    # For simplicity, we draw all nodes as rectangles with text.
    for node_id, (x, y) in pos.items():
        label = G.nodes[node_id]['label']
        # Draw a rectangle
        bbox = dict(boxstyle="round,pad=0.3", fc=node_colors[node_id], ec="black")
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                bbox=bbox, zorder=2)

    plt.title(title)
    plt.tight_layout()
    plt.show()



# Dataset loading
def load_default_dataset():
    """Return the classic 'Play Tennis' dataset with numeric columns mixed in."""
    data = pd.DataFrame([
        # Outlook, Temp(°C), Humidity(%), Wind, PlayTennis
        ['Sunny', 30, 85, 'Weak', 'No'],
        ['Sunny', 30, 90, 'Strong', 'No'],
        ['Overcast', 30, 85, 'Weak', 'Yes'],
        ['Rain', 20, 70, 'Weak', 'Yes'],
        ['Rain', 10, 65, 'Weak', 'Yes'],
        ['Rain', 10, 70, 'Strong', 'No'],
        ['Overcast', 10, 65, 'Strong', 'Yes'],
        ['Sunny', 20, 85, 'Weak', 'No'],
        ['Sunny', 10, 70, 'Weak', 'Yes'],
        ['Rain', 20, 70, 'Weak', 'Yes'],
        ['Sunny', 20, 70, 'Strong', 'Yes'],
        ['Overcast', 20, 85, 'Strong', 'Yes'],
        ['Overcast', 30, 70, 'Weak', 'Yes'],
        ['Rain', 20, 85, 'Strong', 'No']
    ], columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])
    return data

def main():
    # Use command‑line argument or default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        data = pd.read_csv(file_path)
        target_name = data.columns[-1]           # assume last column is target
        feature_names = list(data.columns[:-1])
        # Guess types: if column contains strings/objects -> categorical, else numeric
        feature_types = {}
        for col in feature_names:
            if data[col].dtype == object or data[col].dtype == 'category':
                feature_types[col] = 'categorical'
            else:
                # attempt numeric
                try:
                    pd.to_numeric(data[col])
                    feature_types[col] = 'numeric'
                except ValueError:
                    feature_types[col] = 'categorical'
    else:
        data = load_default_dataset()
        target_name = 'PlayTennis'
        feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
        # Explicitly define types for the default dataset
        feature_types = {
            'Outlook': 'categorical',
            'Temperature': 'numeric',
            'Humidity': 'numeric',
            'Wind': 'categorical'
        }

    print("Dataset preview:")
    print(data.head(), "\n")

    # Build the tree
    tree = build_tree(data, target_name, feature_names, feature_types)

    # Text output
    print("===== Decision Tree (text) =====")
    print_tree(tree, feature_names, target_name)
    print("\n")

    # Graphical output
    try:
        plot_tree(tree, feature_names, target_name, title="Decision Tree (Information Gain)")
    except ImportError as e:
        print("Graphical plot requires networkx and matplotlib.")
        print("Install them with: pip install networkx matplotlib")
        print("Falling back to text tree only.")
        print(e)

if __name__ == "__main__":
    main()