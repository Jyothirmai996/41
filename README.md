# Deep Learning for Personalized Recommendations in Streaming Services
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Simulated dataset generation (user_id, item_id, rating)
def generate_synthetic_data(num_users, num_items, num_samples):
    users = np.random.randint(0, num_users, num_samples)
    items = np.random.randint(0, num_items, num_samples)
    ratings = np.random.randint(1, 6, num_samples)  # Ratings between 1 and 5
    return users, items, ratings

# Parameters
num_users = 1000
num_items = 500
embedding_dim = 50
num_samples = 100000

# Generate synthetic data
users, items, ratings = generate_synthetic_data(num_users, num_items, num_samples)

# Train-test split
X = np.stack([users, items], axis=1)
y = ratings
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
user_input = Input(shape=(1,), name="user_input")
item_input = Input(shape=(1,), name="item_input")

# User embedding
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name="user_embedding")(user_input)
user_vector = Flatten()(user_embedding)

# Item embedding
item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name="item_embedding")(item_input)
item_vector = Flatten()(item_embedding)

# Concatenate embeddings
concatenated = Concatenate()([user_vector, item_vector])

# Dense layers
x = Dense(128, activation="relu")(concatenated)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="linear", name="rating_output")(x)

# Build model
model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# Train the model
history = model.fit(
    [X_train[:, 0], X_train[:, 1]],
    y_train,
    batch_size=128,
    epochs=10,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Make predictions
user_id = 42  # Example user
item_ids = np.arange(num_items)  # All items
predicted_ratings = model.predict([np.full_like(item_ids, user_id), item_ids])
recommended_items = np.argsort(predicted_ratings[:, 0])[-10:][::-1]  # Top 10 items

print(f"Recommended items for user {user_id}: {recommended_items}")
