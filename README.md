# AI Chatbot in Python

## Overview

This is a simple neural network-based chatbot created by following a tutorial. The chatbot can understand user inputs, classify them into different intents, and respond accordingly. 
It includes basic functionality for greeting users, answering programming-related questions, providing learning resources, and even simulating a stock portfolio check.

## Features

- Intent classification using a neural network
- Natural language processing with NLTK (tokenization and lemmatization)
- Customizable intents through a JSON file
- Function mapping for specific intents (like stock checking)
- Training and saving model capabilities

## Requirements

- Python 3.x
- Required packages:
  - `nltk`
  - `numpy`
  - `torch`

Install dependencies with:
```bash
pip install nltk numpy torch
```

## Usage

1. Edit the `intents.json` file to add/modify intents and responses
2. Run the chatbot:
```bash
python main.py
```
3. Type your messages to interact with the chatbot
4. Type `/exit` to quit

## Training the Model

The model is automatically trained when you run `main.py`. Training parameters (batch size, learning rate, epochs) can be adjusted in the `train_model` call.

## Credits

This project was created by following the tutorial:  
**"Advanced AI Chatbot in Python - PyTorch Tutorial"** by NeuralNine  
YouTube Link: [https://www.youtube.com/watch?v=2cq6ywItEv0](https://www.youtube.com/watch?v=2cq6ywItEv0)

Special thanks to NeuralNine for the excellent tutorial that helped me learn about chatbot development with Python and PyTorch.

## Future Improvements

- Add more intents and responses (Probably the best one)
- Improve the neural network architecture
- Add more function mappings
- Implement a GUI interface
- Add persistence for conversation history
- Maybe find a way to feed WhatsApp conversations to train it to understand social media conversations?
