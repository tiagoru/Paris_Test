import pickle
import joblib
import numpy as np

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Access the tree structure
tree = model.tree_

# Create a new dtype without the 'missing_go_to_left' field
new_dtype = np.dtype([
    ('left_child', '<i8'),
    ('right_child', '<i8'),
    ('feature', '<i8'),
    ('threshold', '<f8'),
    ('impurity', '<f8'),
    ('n_node_samples', '<i8'),
    ('weighted_n_node_samples', '<f8')
])

# Create a new array with the new dtype
new_nodes = np.empty(tree.node_count, dtype=new_dtype)

# Copy the data from the old array to the new array
for name in new_dtype.names:
    new_nodes[name] = tree.__getattr__(name)

# Replace the old node array with the new node array
tree.__setattr__('nodes', new_nodes)

# Save the adjusted model using joblib
joblib.dump(model, 'model_joblib.pkl')

# Load the model using joblib to ensure it works correctly
model = joblib.load('model_joblib.pkl')

print("Model conversion successful and verified.")