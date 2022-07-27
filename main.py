import pickle

if __name__ == '__main__':
    # Read and deserialize user_data object from pickle file
    current_usr_data = pickle.load(open('Stanley_usr_data', 'rb'))  # reading binary
    print(f'User preferred 401k type: {current_usr_data.account_401k_type}')
    print(f'Is user over 50: {current_usr_data.over50}')
    print(f'Intent flow for user: \n {current_usr_data.dialog_intents}')
    print(f'Dialog Ouput from User session: \n {current_usr_data.dialog_outputs}')
    print(f'User input data: \n {current_usr_data.usr_inputs}')


