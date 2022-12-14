# Project 2 - Chatbot Project
# execute chat file - contains logic for performing chat with end user
import sys
import pickle
from user_data import UserData
import torch
from bot_model import BotNN
from nlp_utils import make_word_bag, tokenize
import random
import json


device = torch.device('cpu')  # no gpu on my machine

# Loading json file containing intents - essentially the static knowledge base from which chatbot selects replies
with open('chat_intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Loading file containing saved model generated by running train_model.py
file_name = "401k_bot_data.pth"
model_data = torch.load(file_name)

# Assigning
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data["model_state"]

# Load data file saved from training
model = BotNN(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

chatbot_name = "401K Bot"
user_name = input('What is your name? ')

current_conversation_intents = []
current_conversation_outputs = []
current_conversation_inputs = []

# Instantiate object storing user model
current_user_data = UserData(name=user_name, account_401k_type=None, chat_session=None, dialog_flow=None,
                             age_range=None, dialog_intents=current_conversation_intents,
                             dialog_outputs=current_conversation_outputs, usr_inputs=current_conversation_inputs)


print(f'Hi {user_name}, lets have a chat! Type exit to quit')

# Running in a forever loop until user provides input to exit or goodbye intent detected
while True:

    input_text = input("You: ")
    current_user_data.usr_inputs.append(input_text.replace('You:', ''))
    if input_text == "exit":
        # Pickle user data_object to save user/conversation details
        pickle.dump(current_user_data, open(current_user_data.name + '_usr_data', 'wb'))  # writing binary
        break

    # Tokenizing the user input
    input_tokens = tokenize(input_text)

    # Converting user input into bag of words vector
    X = make_word_bag(input_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # retrieve  the best matching predicted intent
    current_tag = tags[predicted.item()]
    # print(f'PREDICTED TAG: {current_tag}')

    # Determine probability scores for prediction
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Required probability score for output can be adjusted if deemed too stringent
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if current_tag == intent["tag"]:
                # Update user model with current intent
                current_user_data.dialog_intents.append(current_tag)
                chat_output = random.choice(intent['responses'])

                # Try getting another random response if current selection already seen in conversation history
                if chat_output in current_conversation_outputs:
                    chat_output = random.choice(intent['responses'])

                # Update output history stored in user model
                current_user_data.dialog_outputs.append(chat_output)

                # Replacing placeholders in json file with personalized name from user model
                if '*NAME*' in chat_output:
                    chat_output = chat_output.replace('*NAME*', current_user_data.name)
                    print(f"{chatbot_name}: {chat_output}")

                    # Pickle user data_object to save user/conversation details
                    pickle.dump(current_user_data, open(current_user_data.name + '_usr_data', 'wb'))  # writing binary
                    sys.exit()

                # Attempt to get user data for certain intents to enable more personalized responses
                print(f"{chatbot_name}: {chat_output}")
                if current_tag == 'roth 401k definition' or current_tag == 'roth benefits':
                    if not current_user_data.account_401k_type:
                        roth_pref = input(f'{current_user_data.name}, do you prefer the roth contributions over '
                                          f'traditional? ')
                        if roth_pref.lower() == 'yes':
                            current_user_data.account_401k_type = 'roth'

                if current_tag == 'traditional 401k definition' or current_tag == 'traditional benefits':
                    if not current_user_data.account_401k_type:
                        traditional_pref = input(
                            f'{current_user_data.name}, do you prefer the traditional account type '
                            f'over roth? ')
                        if traditional_pref.lower() == 'yes':
                            current_user_data.account_401k_type = 'traditional'

                if current_tag == 'contribution limits':
                    age_range = input(f'{current_user_data.name}, are you over 50 years of  age?')
                    if age_range.lower() == 'yes':
                        current_user_data.over50 = True
                        print('Those aged 50 and over can make a $6,500 catch-up contribution in 2021 and 2022')

    else:
        print(f"{chatbot_name}: I did not comprehend that, please try asking differently...")


