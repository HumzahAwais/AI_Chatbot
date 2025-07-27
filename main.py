# Imports
import os # To check if file exists
import json # To work with JSON files
import random # To pick a random response

import nltk # For tokenization and Lemmatization
# Tokenization - splits text into words and phrases
# Lemmatization - reduces words to their base form e.g. "running" to "run"

import numpy as np # For numbers lol

import torch # For deep learning
import torch.nn as nn # For neural networks
import torch.nn.functional as F # For activation functions
import torch.optim as optim # For optimisation lol
from torch.utils.data import DataLoader, TensorDataset # For data loading

nltk.download('wordnet')
nltk.download('punkt')

# Chatbot Model class
class ChatbotModel(nn.Module):
    # Constructor
    def __init__(self, input_size, output_size):
        # Call the parent class constructor
        super(ChatbotModel, self).__init__()

        # Neural network layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(0.5)
    
    # Forward function
    # Forward propagation - when we get an input, how do we get the output?
    def forward(self, x):
        # Feed to fully connected layer 1
        # then apply ReLU activation function
        # then apply dropout
        x = self.dropout(self.relu(self.fc1(x)))

        # Repeat for the second layer
        x = self.dropout(self.relu(self.fc2(x)))

        # Repeat for the third layer
        # But don't apply ReLU or dropout
        x = self.fc3(x)

        return x  # Return the output of the third layer

# Chatbot Assistant class
# This will handle the chatbots functionality
class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        # Tokenize the text
        words = nltk.word_tokenize(text)
        # Lemmatizing the words
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words
    
    # Have a list of words and phrases
    # Mark them as 1 if they are present in the input
    # and 0 if they arent
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        # Check if the intents file exists
        if os.path.exists(self.intents_path):

            # Load the intents from the JSON file
            with open(self.intents_path, 'r') as file:
                intents_data = json.load(file)

            # iterate through each intent
            for intent in intents_data['intents']:
                # Check if tag and responses are present
                # add them if not
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']
                
                # Create numerical patterns
                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))
                
                # Eliminate duplicates using set
                # sort the vocabulary too
                self.vocabulary = sorted(set(self.vocabulary))

    # prepare data function that returns data for neural network training 
    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0] # This is the pattern words
            bag = self.bag_of_words(words) # Create numerical bag of words
            intent_index = self.intents.index(document[1]) # Get the index of the intent tag
            bags.append(bag)
            indices.append(intent_index)

        # Create numpy arrays
        self.X = np.array(bags)
        self.y = np.array(indices)

    # Train the model
    # batch_size: Number of instances to process in parallel at once
    # lr: learning rate which is how quickly the model will move in the direction of the steepest descent
    # epochs: Number of times the model will see the same data
    def train_model(self, batch_size, lr, epochs):
        # Create tensors from the data
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        # Create a dataset based on the tensors
        dataset = TensorDataset(X_tensor, y_tensor)

        # Create a DataLoader to load the dataset
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create model
        # input_size is the size of our X values
        # output_size is the number of intents
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss() # Loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr) # Optimizer

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                # Set all gradients to zero
                optimizer.zero_grad()

                # Forward propagation
                # Get the model outputs for the batch
                outputs = self.model(batch_X)

                # Calculate loss from those outputs
                loss = criterion(outputs, batch_y)

                # Backward propagation
                # Calculate gradients to get direction of improvement
                loss.backward()

                # Update weights
                # Bigger learning rate means bigger steps
                # Smaller learning rate means smaller steps
                optimizer.step()

                # Keep track of total loss
                running_loss += loss.item()#
        
            # Print loss for the epoch
            print(f'Epoch {epoch+1} out of {epochs}, Loss: {running_loss/len(loader):.4f}')

    # Save the trained model to a file
    # model_path = where to save the model
    # dimensions_path = where to save the dimensions of the model
    def save_model(self, model_path, dimensions_path):
        # Save the model state dictionary to the model path
        torch.save(self.model.state_dict(), model_path)

        # Save the input and output sizes to a JSON file (dimensions_path)
        with open(dimensions_path, 'w') as f:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents)
            }, f)
        print(f'Model saved to {model_path} and dimensions saved to {dimensions_path}')

    # Load the model from files
    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            # Load the dimensions of the model
            dimensions = json.load(f)

        # Create model instance with the loaded dimensions
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])

        # Load the model state dictionary
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    # Answer a request
    def process_message(self, input_message):
        # Tokenize and lemmatize the input message
        words = self.tokenize_and_lemmatize(input_message)

        # Create a bag of words from the input message
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        # Evaluate model
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Get predictions from the model
            # The result is a softmax output
            predictions = self.model(bag_tensor)

        # Get the index with the highest probability
        predicted_class_index = torch.argmax(predictions, dim=1).item()

        # The actual class
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                # Call the function mapped to the intent
                self.function_mappings[predicted_intent]()
        
        # Check if the predicted intent has responses
        if self.intents_responses[predicted_intent]:
            # Give a random response from the intent's responses
            return random.choice(self.intents_responses[predicted_intent])
        else:
            # If no responses are available, return a default message
            return "I'm sorry, I don't have a response for that."


def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']
    return random.sample(stocks, 3)

if __name__ == "__main__":
    # Create an instance of the ChatbotAssistant
    assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})

    # Parse intents from the JSON file
    assistant.parse_intents()

    # Prepare data for training
    assistant.prepare_data()

    # Train the model with specified parameters
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)

    # Save the trained model and dimensions
    assistant.save_model('chatbot_model.pth', 'dimensions.json')

    # Using the model
    while True:
        msg = input("You > ")
        if msg.lower() == '/exit':
            print("##### Exiting the chatbot. Goodbye! #####")
            break
        response = assistant.process_message(msg)
        print(f"Bot > {response}")