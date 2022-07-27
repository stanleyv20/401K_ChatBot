# Project 2 - Chatbot Project
# user_data.py - class representing user information generated during chat session

class UserData:
    def __init__(self, name, account_401k_type, chat_session, dialog_flow, age_range, dialog_intents, dialog_outputs,
                 usr_inputs):
        self.name = name
        self.account_401k_type = account_401k_type
        self.chat_session = chat_session
        self.dialog_flow = dialog_flow
        self.over50 = age_range
        self.dialog_intents = dialog_intents
        self.dialog_outputs = dialog_outputs
        self.usr_inputs = usr_inputs
