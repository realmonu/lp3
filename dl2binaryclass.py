import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers, callbacks
import tensorflow as tf
import random

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ==============================
# PART 1: Model Building and Training
# ==============================

# Load the dataset with a vocabulary size limit
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences to ensure uniform input length
max_sequence_length = 200
X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

# Convert labels to NumPy arrays (optional, as Keras handles lists)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Build the model
def build_model():
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))
    model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = build_model()

# Add early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate on the test set
print("\nEvaluating the model on the test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

# ==============================
# PART 2: Final Test Evaluation
# ==============================

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# ==============================
# PART 3: Decoding and Predicting Sample Reviews
# ==============================

# Load the IMDb word index for decoding
word_index = imdb.get_word_index()

# Function to decode encoded reviews into words
def decode_review(encoded_review):
    reverse_word_index = {value + 3: key for (key, value) in word_index.items()}  # Add offset for special indices
    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<UNK>'
    reverse_word_index[3] = '<UNUSED>'
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

# Sample some reviews for prediction
print("\nPredicting sentiment for sample reviews...")
sample_indices = random.sample(range(len(X_test)), 3)
sample_reviews = [decode_review(X_test[i]) for i in sample_indices]

# Preprocess the reviews by converting them to sequences of indices
preprocessed_reviews = pad_sequences(
    [[word_index.get(word, 2) for word in review.split()] for review in sample_reviews],
    maxlen=200
)

# Make predictions on the sample reviews
predictions = model.predict(preprocessed_reviews)

# Print the results with actual and predicted sentiments
for i, (review, prediction, actual) in enumerate(zip(sample_reviews, predictions, [y_test[i] for i in sample_indices])):
    print(f'\nReview {i+1}:\n"{review}"')
    print(f'Predicted Sentiment: {"Positive" if prediction > 0.5 else "Negative"}')
    print(f'Actual Sentiment: {"Positive" if actual == 1 else "Negative"}\n')
