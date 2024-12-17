import torch

#url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#response = requests.get(url)

'''with open('sample.txt', 'wb') as file:
    file.write(response.content) # write the content of the response to a new file
    '''
    
print('File downloaded')
#read the data
with open('frist-names.txt', 'r') as file:
    text = file.read()
    
    # Split text into words and create vocabulary
    words = text.split()
    vocab = sorted(set(words))

    # Create mapping from words to integers and vice versa
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    # Convert text to tensor of indices
    encoded = torch.tensor([word_to_ix[word] for word in words], dtype=torch.long)
    decoded = ' '.join([ix_to_word[ix.item()] for ix in encoded])
    
    #print('encoded text: ', encoded[:100], 'decoded text: ')
    
    # Split data into training and validation sets
    train_data = encoded[:int(0.9*len(encoded))]
    val_data = encoded[int(0.9*len(encoded)):]
    
    # Batch the data
    torch.manual_seed(1337)
    
    sequence_size = 32
    batch_size = 32
    
    def get_batches(data, sequence_size):
        ix = torch.randint(len(data)-sequence_size, (batch_size,))
        
        x = torch.stack([data[i:i+sequence_size] for i in ix])
        y = torch.stack([data[i+1:i+1+sequence_size] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    import torch.nn as nn
    
    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dropout=0.1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = nn.Embedding(sequence_size, d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(d_model, vocab_size)
            
            self.d_model = d_model
            
        def forward(self, x):
            # Create position indices
            positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
            
            # Combine token embeddings and positional encodings
            x = self.embedding(x) + self.pos_encoder(positions)
            
            # Transformer expects shape (seq_len, batch, d_model)
            x = x.transpose(0, 1)
            x = self.transformer(x)
            x = x.transpose(0, 1)
            
            return self.fc_out(x)

    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(len(vocab)).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    
    epochs = 200
    
    for epoch in range(epochs):
        model.train()
        x, y = get_batches(train_data, sequence_size)
        optimizer.zero_grad()

        # Forward pass
        output = model(x)
        loss = criterion(output.view(-1, len(vocab)), y.view(-1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            
# Generate text
def generate_text(model, start_text, max_length=50):
    model.eval()
    words = start_text.split()
    encoded = [word_to_ix[word] for word in words]
    inputs = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            if inputs.size(1) > sequence_size:
                inputs = inputs[:, -sequence_size:]
            
            output = model(inputs)
            _, predicted = torch.max(output[:, -1:], 2)
            inputs = torch.cat([inputs, predicted], 1)
            
    return ' '.join([ix_to_word[ix.item()] for ix in inputs[0]])

print(generate_text(model, 'the meaning of life is, something fun', max_length=1000))