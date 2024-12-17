import requests
import torch

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
response = requests.get(url)

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

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, sequence_size, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x.transpose(0,1))
        x = self.fc_out(x).transpose(0,1)
        return x

model = TransformerModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    x_batch, y_batch = get_batches(train_data, sequence_size)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs.reshape(-1, vocab_size), y_batch.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    scheduler.step(loss)
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

def generate_text(model, start_text, max_length=500):
    model.eval()
    chars = list(start_text)
    encoded = encode(chars)
    inputs = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            if inputs.size(1) > sequence_size:
                inputs = inputs[:, -sequence_size:]
            
            output = model(inputs)
            next_char_logits = output[0, -1, :]
            probs = F.softmax(next_char_logits, dim=0)
            next_char_index = torch.multinomial(probs, 1)
            
            inputs = torch.cat([inputs, next_char_index.unsqueeze(0)], dim=1)
            
            if index_to_char[next_char_index.item()] == '\n':
                break
    
    generated = decode(inputs[0].tolist())
    return generated

# Generate some text
print(generate_text(model, "The ", 500))
