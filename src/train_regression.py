import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from gnn import GNNShortestPathLength
from data_utils import generate_dataset

# Parameters
dataset_size = 500
num_nodes = 6
num_edges = 10
max_path_length = 10
batch_size = 32
epochs = 20

# Generate dataset
data = generate_dataset(num_graphs=dataset_size, num_nodes=num_nodes, num_edges=num_edges, max_path_length=max_path_length, task='regression')
X, A, Y = zip(*data)
X = np.array(X)
A = np.array(A)
Y = np.array(Y)

# Train/test split
X_train, X_test, A_train, A_test, Y_train, Y_test = train_test_split(X, A, Y, test_size=0.2, random_state=42)

# Build model
model = GNNShortestPathLength(n_hidden=32)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit([X_train, A_train], Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate
test_loss, test_mae = model.evaluate([X_test, A_test], Y_test)
model.save_weights('regression_model.h5')
print(f"Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}") 