import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Parameters
num_sources = 4
input_dim = 10
hidden_units = 640
learning_rate = 0.001
num_epochs = 500
batch_size = 32
alpha = 10  # Lagrangian penalty coefficient

# Generate synthetic data with non-linear pattern
def generate_data(num_samples=1000, num_sources=4, input_dim=10):
    X = np.random.rand(num_samples, input_dim)
    sources = []
    for i in range(num_sources):
        pattern = np.sin(np.pi * X @ np.random.rand(input_dim, 1)) + np.random.normal(0, 0.1, (num_samples, 1))
        sources.append(pattern)
    sources = np.hstack(sources)
    return X, sources

# Improved Model Definition
class MultiSourceModel(tf.keras.Model):
    def __init__(self, num_sources, input_dim, hidden_units):
        super(MultiSourceModel, self).__init__()
        self.num_sources = num_sources
        
        # More complex hidden layers
        self.hidden = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units // 2, activation='relu'),
            tf.keras.layers.Dense(hidden_units // 4, activation='relu')
        ])
        
        # Output layer with activation
        self.output_layer = tf.keras.layers.Dense(1, activation='tanh')
        
        # Dynamically learnable lambdas using softmax to ensure sum to 1
        self.lambda_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_sources, activation='softmax')
        ])
    
    def call(self, inputs):
        x = self.hidden(inputs)
        output = self.output_layer(x)
        # lambdas = self.lambda_net(inputs) # use either softmax to scale to 1 or the lagrangian
        return output, lambdas

# Custom Loss Function
def lagrangian_loss(pred, lambdas, sources, model):
    # Weighted sum of sources to form "pseudo ground truth"
    pseudo_ground_truth = tf.reduce_sum(sources * lambdas, axis=-1)
    
    # Mean Squared Error Loss
    mse_loss = tf.reduce_mean(tf.square(pred - pseudo_ground_truth))
    
    # Lagrangian penalty to enforce sum of lambdas = 1
    lagrangian_penalty = alpha * tf.square(tf.reduce_sum(lambdas, axis=-1) - 1.0)
    
    return mse_loss + tf.reduce_mean(lagrangian_penalty)

# Training Step
@tf.function
def train_step(model, optimizer, x_batch, sources_batch):
    with tf.GradientTape() as tape:
        pred, lambdas = model(x_batch)
        loss = lagrangian_loss(pred, lambdas, sources_batch, model)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Prepare Data
X, sources = generate_data(num_sources=num_sources, input_dim=input_dim)
X_train, X_test, sources_train, sources_test = train_test_split(X, sources, test_size=0.2, random_state=42)

# Initialize Model and Optimizer
model = MultiSourceModel(num_sources, input_dim, hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i+batch_size]
        sources_batch = sources_train[i:i+batch_size]
        sources_batch = tf.cast(sources_batch, tf.float32)
        
        loss = train_step(model, optimizer, x_batch, sources_batch)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy()}")

# Check learned lambdas
_, lambdas = model(X_test)
print("Learned lambdas (sample):", lambdas.numpy()[:5])

# Recursive Prediction Testing Phase
val = X_test[0:1]  # Start with the first test sample
predictions = []

for _ in range(5):
    pred, _ = model(val)
    predictions.append(pred.numpy().flatten()[0])  # Save prediction
    
    # Repeat the prediction to match input shape (1, input_dim)
    val = np.repeat(pred.numpy().flatten()[0], input_dim).reshape(1, -1)
    val = tf.cast(val, tf.float32)  # Ensure consistent data type

print("Recursive Predictions:", predictions)

