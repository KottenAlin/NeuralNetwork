from TransformerModel import TransformerLanguageModel, decode, encode, plot_model, plot_training
import TransformerModel
import torch
import torch.nn as nn
import SimpleNeuralNetwork as SNN
import BigramModel

#clear the screen
import os
from colorama import init, Fore
import sys
import glob
init()  # Initialize colorama

# Alternative way to clear screen
def clear_screen():
    print('\033[H\033[J')

clear_screen()

def run_snn():
    # Run the simple neural network
    print("Arguments for main function:")
    for arg in sys.argv:
        print(arg)
    
    SNN.main()

def load_model(model_path):
    # Load the trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the mapping
    stoi = torch.load(os.path.join(model_path, 'stoi.pth'), weights_only=True)
    itos = torch.load(os.path.join(model_path, 'itos.pth'), weights_only=True)
    vocab_size = torch.load(os.path.join(model_path, 'vocab_size.pth'), weights_only=True)
    
    model = TransformerLanguageModel(vocab_size)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))  # Load from specified directory
    model = model.to(device)
    
    return model, stoi, itos, device
    
def print_start_menu():
    menu = f"""
    {Fore.CYAN}=============================
    Welcome human to the Chatbot interface! Feel free to chat with me or train a new model.
    =============================
    1. Start Chat
    2. Plot Models
    3. Run Simple Neural Network
    4. Train New Model
    5. Run Bigram Model
    6. exit
    ============================={Fore.RESET}
    """
    print(menu)
    
def print_plot_menu():
    #clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    menu = """
    =============================
Type the name of the model you want to plot:
    =============================
    """
    print(menu)
    list_language_models()
    print('==================================')
    choice = input("command:")
    
    if choice in ('exit', 'quit'):
        exit()
    else:
        try:
            model, stoi, itos, device = load_model('LanguageModels/' + choice + '/')
            
            plot_model(model)
            
            # load losses and validation losses
            losses = torch.load(os.path.join('LanguageModels/' + choice + '/', 'losses.pth'), weights_only=True)
            val_losses = torch.load(os.path.join('LanguageModels/' + choice + '/', 'val_losses.pth'), weights_only=True)
            
            plot_training(losses, val_losses)
        except FileNotFoundError:
            print(f"The model '{choice}' does not exist.")
        input("Press Enter to continue...")
    
    os.system('cls' if os.name == 'nt' else 'clear')
    input("Press Enter to continue...")
    exit()
    
def list_language_models():
    path = 'LanguageModels/'
    if os.path.exists(path):
        folders = [f.name for f in os.scandir(path) if f.is_dir()]
        for folder in folders:
            print(f"{folders.index(folder) + 1}. {folder}")
    else:
        print(f"The directory '{path}' does not exist.")

def chat():
    
    #clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print ("=============================")
    print ("Type the name of the model you want to chat with")
    print ("=============================")
    list_language_models()
    print('===============================')
    choice = input("command:")
    if choice in ('exit', 'quit'):
        exit()
    print('===============================')
    embedding_level = input("Enter the embedding level ('word' or 'char'): ")

    try:
        model, stoi, itos, device = load_model('LanguageModels/' + choice + '/')
        #clear the screen
        os.system('cls' if os.name == 'nt' else 'clear')
        #max_tokens = int(input("Enter the maximum number of tokens to generate: "))
        #temperature = int(input("Enter the temperature: "))
        os.system('cls' if os.name == 'nt' else 'clear')
        max_tokens = int(input("Enter the maximum number of tokens to generate (default=50): "))
        chat_with_model(model, stoi, itos, device, embedding_level, max_tokens)
    except FileNotFoundError:
        print(f"The model '{choice}' does not exist.")
        input("Press Enter to continue...")
    
def chat_with_model(model, stoi, itos, device, embedding_level,  max_tokens=50, temperature=1):
        while True:
            context = input("You: ")
            if context.lower() in ('exit', 'quit'):
                break
            print(Fore.RED + context + Fore.RESET)

            context_tensor = torch.tensor(encode(context, stoi, embedding_level), dtype=torch.long, device=device).unsqueeze(0)

            generated_text = decode(model.generate(context_tensor, max_tokens, temperature)[0].tolist(), itos, embedding_level)
            print(Fore.GREEN + generated_text + Fore.RESET)

def train_new_model():
    print("====================================")
    print("Set training data")
    print("====================================")
    data_files = glob.glob('Data/*') # Get all files in the Data directory,
    for file in data_files:
        print(file)
    
    data_file = input("Enter the name of the data file: ")
    
    os.system('cls' if os.name == 'nt' else 'clear')
    # Load and print global variables from bettertransform
    print("Hyperparameters")
    print("====================================")
    for name in dir(bettertransform):
        if not name.startswith("__"):
            print(name)

    try:
        with open(data_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"The file '{data_file}' does not exist.")
        input("Press Enter to continue...")
        return
    print("====================================")
    encoding_level = input("Enter the encoding level ('word' or 'char'): ")
    stoi, itos, vocab_size = bettertransform.generate_mapping(text, encoding_level)
    print("====================================")
    print('vocab_size:', vocab_size)
    print("====================================")
    model_name = input("Enter the name of the model: ")
    os.system('cls' if os.name == 'nt' else 'clear')
    
    model_dir = 'LanguageModels/' + model_name + '/'
    #create the directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        print(f"The model '{model_name}' already exists.")
        input("Press Enter to continue...")
        return
        
    # Save the mapping
    torch.save(stoi, os.path.join(model_dir, 'stoi.pth'))
    torch.save(itos, os.path.join(model_dir, 'itos.pth'))
    torch.save(vocab_size, os.path.join(model_dir, 'vocab_size.pth'))
    
    model = TransformerLanguageModel(vocab_size)
    
    train_data, val_data = bettertransform.generate_data(text, stoi)
    
    iterations = int(input("Enter the number of iterations: "))
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    
    losses, val_losses = model.train(train_data, val_data, iterations)
    os.system('cls' if os.name == 'nt' else 'clear')
    plot_training(losses, val_losses)
    
    #print final loss
    print('====================================')
    print(f"Final loss: {losses[-1]}")
    print('====================================')
    
    #save the losses and val_losses
    torch.save(losses, os.path.join(model_dir, 'losses.pth'))
    torch.save(val_losses, os.path.join(model_dir, 'val_losses.pth'))
    
    #save the model
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    try:
        print_start_menu()
        choice = input("command:")
        
        if choice == '1':
            chat()
        elif choice == '2':
            print_plot_menu()
        elif choice == '3':
            run_snn()
        elif choice == '4':
            # Train model
            os.system('cls' if os.name == 'nt' else 'clear')
            train_new_model()
        elif choice == '5':
            BigramModel.main()
            print("================================")
            input("Press Enter to continue...")

        elif choice == '6' or choice.lower() == 'exit':
            exit()
        else:
            print("Invalid command")
            input("Press Enter to continue...")
        
        main()
    except KeyboardInterrupt:
        exit()
    # Generate text
    #context = input("Enter a starting word: ")

if __name__ == '__main__':
    main()