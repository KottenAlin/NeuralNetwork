import requests
import torch

#url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#response = requests.get(url)

'''with open('sample.txt', 'wb') as file:
    file.write(response.content) # write the content of the response to a new file
    '''
    
print('File downloaded')
#read the data
with open('sample.txt', 'r') as file:
    text = file.read()
#print("length of text in characters: ", len(text))
#print(text[:1000])

#get all the unique characters in the file
vocab = sorted(set(text)) #use the set function to get all the unique characters in the text, then sort the
vocab_size = len(vocab)

#tokenize the text
char_to_index = {char:index for index, char in enumerate(vocab)} #create a dictionary that maps each unique character to a unique index
index_to_char = {index:char for char, index in char_to_index.items()} #create a dictionary that maps each unique index to a unique character

def encode(text):
    return [char_to_index[char] for char in text] #convert the text to a list of indices that maps to some character, using the char_to_index dictionary

def decode(indices):
    return ''.join([index_to_char[index] for index in indices]) #convert the list of indices back to text, using the index_to_char dictionary

#print('encoded text: ', encode(text[:100]), 'decoded text: ', decode(encode(text[:100])))


#encode the entire dataset
encoded_text = encode(text)
#turn this into a pytorch dataset
data = torch.tensor(encoded_text, dtype=torch.int64) #convert the encoded text to a tensor
#split the data into training and validation sets
train_data = data[:int(0.9*len(data))] #90% of the data for training
val_data = data[int(0.9*len(data)):] #10% of the data for validation

#batch the data
torch.manual_seed(1337) # Set seed for reproducibility

sequence_size = 64 # the length of charcters in the sequence that we will use to make the prediction
batch_size = 32 # the number of sequences in a batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #check if cuda is available

def get_batches(data, sequence_size): #function to get the batches of data and the corresponding ouput sequences (with 1 extra character)
    ix =torch.randint(len(data)-sequence_size, (batch_size,)) #get a random starting index for each sequence in the batch
    
    x = torch.stack([data[i:i+sequence_size] for i in ix]) #get the sequences, and stack them into a tensor of shape (batch_size, sequence_size)
    y = torch.stack([data[i+1:i+1+sequence_size] for i in ix]) #get the target sequences with 1 extra character
    x, y = x.to(device), y.to(device) #move the data to the device 
    return x, y

import torch.nn as nn
from torch.nn import functional as F

class AutoencoderModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=1, nhead=8, lr=0.001):
        super(AutoencoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size) # create an embedding layer with the vocab_size as the input size and hidden_size as the output size
        self.encoder = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True), num_layers),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True), num_layers)
        ) # create the encoder layer with the hidden_size as the input size and add more layers
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(hidden_size, nhead, batch_first=True), num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        
    def forward(self, x, y):
        x_embedded = self.embedding(x)
        y_embedded = self.embedding(y)
        x_encoded = self.encoder(x_embedded)
        y_decoded = self.decoder(y_embedded, x_encoded)
        output = self.fc(y_decoded)
        
        loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1)) #calculate the loss using cross entropy, and flatten the output and target tensors
        return output, loss
    
    def backwardprop(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def train(self, x, y, epochs=10):
        losses = []
        for epoch in range(epochs):
            y_pred, loss = self.forward(x, y)
            self.backwardprop(loss)
            losses.append(loss.item())
            if epoch % 10 == 0:
                #estimated_loss = estimate_loss(self, x, y)
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
                
                #validate the model
                val_x, val_y = get_batches(val_data, sequence_size)
                val_loss = self.forward(val_x, val_y)[1]
                print('Validation loss: ', val_loss.item())
        return losses
    
    def generate_text(self, input_sequence, max_length=50, temperature=1.0):
        generated_sequence = input_sequence
        x_embedded = self.embedding(input_sequence)
        x_encoded = self.encoder(x_embedded)
    
        for _ in range(max_length):
            last_token = generated_sequence[:, -1:]
            y_embedded = self.embedding(last_token)
            y_decoded = self.decoder(y_embedded, x_encoded)
            output = self.fc(y_decoded)
    
            # Apply temperature to the logits
              # You can adjust this value to control the randomness
            logits = output[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token = output.argmax(dim=-1) #get the index of the token with the highest probability
            
            next_token = next_token[:, -1:]  # Ensure the shape is [batch_size, 1]
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
            #generated_text = decode(generated_sequence.view(-1).cpu().numpy())
            #print('generated_sequence',generated_text)
        return generated_sequence

    
    def generate_text_from_text(self, input_text, max_length=100):
        
        input_indices = [self.char_to_index.get(char, 0) for char in input_text]
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.embedding.weight.device)
        generated_indices = input_indices.copy()

        for _ in range(max_length):
            with torch.no_grad():
                embeddings = self.embedding(input_tensor)
                encoded = self.encoder(embeddings)
                decoded = self.decoder(embeddings, encoded)
                logits = self.fc(decoded)
                next_token_logits = logits[:, -1, :]
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_index = torch.multinomial(probabilities, num_samples=1).item()
                generated_indices.append(next_index)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_index]], device=input_tensor.device)], dim=1)

        generated_text = ''.join([self.index_to_char.get(idx, '') for idx in generated_indices])
        
        return generated_text
        
    
nn = AutoencoderModel(vocab_size)

neural_network = nn.to(device)
x, y = get_batches(train_data, sequence_size) #get a sample batch

neural_network.train(x, y, epochs=100)

# validate the model
val_x, val_y = get_batches(val_data, sequence_size)


output, val_loss = neural_network(val_x, val_y)
print('loss', val_loss.item())


'''input_text = 'The meaning of life is'

generated_text = neural_network.generate_text_from_text(input_text, max_length=100)
print('generated_text',generated_text)'''

#generate some text
generated_text = neural_network.generate_text(val_x, max_length=1)
print(generated_text.shape)

generated_text = decode(generated_text.view(-1).cpu().numpy())
print(generated_text)

#save the model
torch.save(neural_network.state_dict(), 'autoencoder_model.pth')
