import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from gnn import GNNShortestPathSequence
from data_utils import generate_dataset

# Parameters
dataset_size = 500
num_nodes = 6
num_edges = 10
max_path_length = 10
batch_size = 32
epochs = 20
pad_value = -1

# Generate dataset
data = generate_dataset(num_graphs=dataset_size, num_nodes=num_nodes, num_edges=num_edges, max_path_length=max_path_length, task='sequence')
X, A, Y = zip(*data)
X = np.array(X)
A = np.array(A)
Y = np.array(Y)

# Mask for padded values
def create_mask(y, pad_value):
    return (y != pad_value)

# Train/test split
X_train, X_test, A_train, A_test, Y_train, Y_test = train_test_split(X, A, Y, test_size=0.2, random_state=42)

# Build model
model = GNNShortestPathSequence(n_hidden=32, max_path_length=max_path_length, num_nodes=num_nodes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare targets for sparse categorical crossentropy (expand dims)
Y_train_exp = np.expand_dims(Y_train, -1)
Y_test_exp = np.expand_dims(Y_test, -1)

# Train
model.fit([X_train, A_train], Y_train_exp, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate
preds = model.predict([X_test, A_test])
preds_indices = np.argmax(preds, axis=-1)

# Exact match accuracy (all steps correct)
exact_match = np.all(preds_indices == Y_test, axis=1)
exact_match_acc = np.mean(exact_match)
print(f"Exact match accuracy: {exact_match_acc:.4f}")

# Per-step accuracy (ignoring pad_value)
mask = (Y_test != pad_value)
per_step_acc = np.sum((preds_indices == Y_test) & mask) / np.sum(mask)
print(f"Per-step accuracy: {per_step_acc:.4f}") 