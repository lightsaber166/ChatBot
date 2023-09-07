import tensorflow as tf
import json
import random

# Load the trained model
model = tf.keras.models.load_model("intent_model_tf.keras")

# Load the tokenizer
with open("tokenizer.json", "r") as tokenizer_file:
    tokenizer_json = tokenizer_file.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Load the intent responses
with open("intents.json", "r") as intents_file:
    intents_data = json.load(intents_file)
    intents = intents_data["intents"]

# Define a function to classify user input and generate responses
def chat():
    print("Chatbot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ").lower()
        if user_input == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break

        # Tokenize the user input
        user_input_seq = tokenizer.texts_to_sequences([user_input])
        user_input_padded = tf.keras.preprocessing.sequence.pad_sequences(user_input_seq, maxlen=model.input_shape[1], padding="post")

        # Predict the intent
        predicted_intent_id = model.predict(user_input_padded).argmax()
        predicted_intent = intents[predicted_intent_id]
        available_responses = predicted_intent["responses"]

        # Get a random response from the predicted intent
        response = random.choice(available_responses)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
