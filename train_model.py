# Project 2 - Chatbot Project
# train_model file - contains logic for generating model using neural net for use in execute_chat.py
from bot_model import BotNN
from nlp_utils import make_word_bag, stem, tokenize
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Loading json file containing intents - essentially the static knowledge base from which chatbot selects replies
with open('chat_intents.json', 'r') as my_file:
    chat_intents = json.load(my_file)

total_vocab = []
intent_tags = []
pattern_tag_pairs = []

# iterating over intents specified in json file and appending to respective arrays
for intent in chat_intents['intents']:
    current_tag = intent['tag']
    intent_tags.append(current_tag)
    for pattern in intent['patterns']:
        pattern_tokens = tokenize(pattern)
        # extending rather than appending to avoid having an array containing arrays
        total_vocab.extend(pattern_tokens)
        # Add to pattern and tag list
        pattern_tag_pairs.append((pattern_tokens, current_tag))

# stem and lower each word
punctuation = ['.', '?', '!']
total_vocab = [stem(w) for w in total_vocab if w not in punctuation] # Ignore punctuation

# removing duplicates using set
total_vocab = sorted(set(total_vocab))  # applying sorted function to generate list from set
tags = sorted(set(intent_tags))

# print(len(pattern_tag_pairs), "patterns")
# print(len(tags), "tags:", tags)
# print(len(total_vocab), "unique stemmed words:", total_vocab)

X_train = []  # Will contain patterns corresponding to intents
y_train = []  # Will contain an index representing the tag of an intent

for (pattern_phrase, tag) in pattern_tag_pairs:
    # generates a vector (bag of words) for each phrase in an available intent specified in json file
    word_bag = make_word_bag(pattern_phrase, total_vocab)
    X_train.append(word_bag)
    label = tags.index(tag)  # Maps each label (tag defined in intents) to a number
    y_train.append(label)  # Might need to use 1 hot encoding, not needed for pytorch because of CrossEntropyLoss

# Converting lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters - adjust as desired
num_epochs = 1000
batch_count = 8
learning_rate = 0.001 # Controls step-size used when updating weights in neural network
input_count = len(X_train[0])  # Will correlate to number of unique tokens in entire vocabulary
hidden_count = 8
# number of output params will correlate to count of available intents
output_count = len(tags)


# Code to enable generation of training data
class BotDataSet(Dataset):
    def __init__(self):
        self.x_data = X_train  # phrases
        self.y_data = y_train  # labels
        self.sample_count = len(X_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.sample_count


chat_bot_data = BotDataSet()
trainer = DataLoader(dataset=chat_bot_data, batch_size=batch_count, shuffle=True, num_workers=0)
device = 'cpu'  # no gpu on my machine
# Instantiating neural network
chatbot_model = BotNN(input_count, hidden_count, output_count).to(device)

# defining hyper parameters
learn_rate = 0.001
# Adjust if suspecting over fitting
epoch_count = 1000

# Setting up optimizer and loss
criterion = torch.nn.CrossEntropyLoss()
# Applying Adam optimization algorithm to lower errors via gradiant descent
# Other optimization algorithms can be viewed here: https://pytorch.org/docs/stable/optim.html
optimizer = torch.optim.Adam(chatbot_model.parameters(), lr=learn_rate)

# Train the model across selected number of epochs
for epoch_instance in range(epoch_count):
    for (words, labels) in trainer:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # performing forward pass through neural net
        outputs = chatbot_model(words)
        loss = criterion(outputs, labels)
        # perform back propagation of error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Printing output to display progress
    if (epoch_instance + 1) % 50 == 0:
        print(f'Training EPOCH: {epoch_instance + 1}/{epoch_count} - loss = {loss.item():.4f}')
print(f'End loss after training = {loss.item():.4f}')

# Creating dictionary to save data on the trained model for use in chat
bot_data_model = {
    "model_state": chatbot_model.state_dict(),
    "input_size": input_count,
    "hidden_size": hidden_count,
    "output_size": output_count,
    "all_words": total_vocab,
    "tags": tags
}

# saving file within current directory for use in chat application
file_name = "401k_bot_data.pth"
torch.save(bot_data_model, file_name)
print(f'ChatBot training complete! Data file saved as: {file_name}')
