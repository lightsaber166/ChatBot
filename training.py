import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the intents from intents.json
intents_path = "intents.json"
with open(intents_path, "r") as file:
    intents_data = json.load(file)

# Prepare training data
texts = []
labels = []

for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# Tokenize and encode labels
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
encoded_labels = LabelEncoder().fit_transform(labels)

# Convert text to sequences and pad them
sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(set(labels)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model with 200 epochs
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))

# Save the model in Keras format
model.save("intent_model_tf.keras")
with open("tokenizer.json", "w") as tokenizer_file:
    tokenizer_json = tokenizer.to_json()
    tokenizer_file.write(tokenizer_json)

print("Model and tokenizer saved in Keras format after 200 epochs.")
