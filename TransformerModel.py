import torch
import torch.nn as nn
from torch.nn import functional as F
from colorama import init, Fore
import seaborn as sns
import torch
import os
init()  # Initialize colorama

# hyperparameters
batch_size = 16 # how many independent sequencesh will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iterations = 2000
eval_interval = 100 # how often to evaluate the model on train and val sets
learning_rate = 1e-3 # how big of a step to take when updating the model weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64 # how many dimensions to use for the embedding and hidden state?
n_head = 4 # how many independent self-attention heads to use?
n_layer = 4 # how many layers in the model
dropout = 0.0 #when training, what fraction of activations to randomly set to zero in the self-attention heads?
# ------------

def print_hyperparameters():
    print(f'batch_size: {batch_size}')
    print(f'block_size: {block_size}')
    print(f'max_iterations: {max_iterations}')
    print(f'eval_interval: {eval_interval}')
    print(f'learning_rate: {learning_rate}')
    print(f'device: {device}')
    print(f'n_embd: {n_embd}')
    print(f'n_head: {n_head}')
    print(f'n_layer: {n_layer}')
    print(f'dropout: {dropout}')

def generate_mapping(text, level='char'):
    """
    Generate mapping from tokens to integers and vice versa.
    level: 'char' for character-level encoding, 'word' for word-level encoding.
    """
    if level == 'char':
        # Character-level mapping
        tokens = sorted(list(set(text)))
    elif level == 'word':
        # Word-level mapping
        tokens = sorted(list(set(text.split())))
    else:
        raise ValueError("Invalid level. Use 'char' or 'word'.")
    vocab_size = len(tokens)
    print('vocabulary size:', vocab_size)
    # Create mappings
    stoi = { token: i for i, token in enumerate(tokens) }
    print(stoi)
    itos = { i: token for i, token in enumerate(tokens) }
    return stoi, itos, vocab_size

def encode(s, stoi, level='char'):
    # Encoder: take a string, output a list of integers
    if level == 'char':
        return [stoi[c] for c in s if c in stoi]
    elif level == 'word':
        return [stoi[word] for word in s.split() if word in stoi]
    else:
        raise ValueError("Invalid level. Use 'char' or 'word'.")

def decode(l, itos, level='char'):
    # Decoder: take a list of integers, output a string
    if level == 'char':
        return ''.join([itos[i] for i in l])
    elif level == 'word':
        return ' '.join([itos[i] for i in l])
    else:
        raise ValueError("Invalid level. Use 'char' or 'word'.")

def generate_data(text, stoi, level='char'):
    # Generate training and validation data
    data = torch.tensor(encode(text, stoi, level=level), dtype=torch.long)
    n = int(0.9 * len(data))  # First 90% for training, rest for validation
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

# data loading
def get_batch(split, train_data, val_data, batch_size=batch_size, block_size=block_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # choose random starting points for each sequence in the batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # x is a batch of blocks of text
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # y is the same batch of blocks, shifted by 1 character
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, seq_len, embed_dim = x.shape # Extract the shape of the input tensor x.
    
        keys = self.key(x)   # (batch_size, seq_len, embed_dim)
        # Apply the key transformation to the input tensor x.
    
        queries = self.query(x)# Apply the query transformation to the input tensor x. This also typically involves a linear transformation.
        # compute attention scores ("affinities")
        attention_scores = queries @ keys.transpose(-2, -1) * embed_dim**-0.5 # Compute the attention scores by performing a matrix multiplication between queries and the transpose of keys.
        # Scale the scores by the square root of the embedding dimension to stabilize gradients.
        attention_scores = attention_scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))# Apply a mask to the attention scores to ensure that the model does not attend to future positions in the sequence.
        attention_probs = F.softmax(attention_scores, dim=-1) # Apply the softmax function to the attention scores to obtain the attention probabilities.
        # This normalizes the scores so that they sum to 1 along the last dimension.

        attention_probs = self.dropout(attention_probs) # Apply dropout to the attention probabilities for regularization.

        # perform the weighted aggregation of the values
        values = self.value(x) # Apply the value transformation to the input tensor x. This typically involves a linear transformation.
        out = attention_probs @ values # Compute the output by performing a matrix multiplication between the attention probabilities and the values.
        # This aggregates the values based on the attention probabilities.

        return out
        # Return the final output tensor.


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # the linear layer that takes the concatenated outputs of all heads and projects it back to n_embd dimensions
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
        


class FeedFoward(nn.Module):
    """linear feedforward layer followed by a ReLU for non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ 1 transformer block = multi-head attention + feedforward with normalization"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_dim = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_dim)  # self-attention
        self.ffwd = FeedFoward(n_embd)  # feed-forward network
        self.ln1 = nn.LayerNorm(n_embd)  # layer normalization after self-attention
        self.ln2 = nn.LayerNorm(n_embd) # layer normalization after feed-forward network

    def forward(self, x):
        # Apply layer normalization and self-attention, then add the result to the input (residual connection)
        x = x + self.sa(self.ln1(x))
        # Apply layer normalization and feed-forward network, then add the result to the input (residual connection)
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the predictions for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, context, targets=None):
        batch_size, seq_len = context.shape
        # Get token embeddings for the input indices
        token_embedding = self.token_embedding_table(context)
        # Get positional embeddings for the sequence positions
        positional_embedding = self.position_embedding_table(torch.arange(seq_len, device=device))
        # Add token and positional embeddings
        x = token_embedding + positional_embedding
        # Pass through transformer blocks
        x = self.blocks(x)
        # Apply final layer normalization
        x = self.ln_f(x)
        # Get predictions from the language model head
        predictions = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # Reshape predictions and targets for calculating loss
            batch_size, seq_len, vocab_size = predictions.shape
            predictions = predictions.view(batch_size * seq_len, vocab_size)
            targets = targets.view(batch_size * seq_len)
            # Calculate cross-entropy loss
            loss = F.cross_entropy(predictions, targets)

        return predictions, loss

    def generate(self, context, max_new_tokens, temperature=1.0):
        # context is a sequence of indices that the model will use to generate the next token
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens
            if context.size(1) <= block_size:
                context_cond = context
            else:
                # crop context to the last block_size tokens
                context_cond = context[:, -block_size:]
            # get the predictions
            predictions, _ = self(context_cond)
            # focus only on the last time step
            predictions = predictions[:, -1, :] 
            # scale predictions by temperature before softmax
            if temperature != 0:
                predictions = predictions / temperature
            # apply softmax to get probabilities
            probs = F.softmax(predictions, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) 
            # append sampled index to the running sequence
            context = torch.cat((context, next_token), dim=1) 
        return context
    
    def train(self, train_data, val_data, max_iterations=max_iterations):
        losses, val_losses = [], []
            # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate) # use AdamW optimizer, which is a modern optimizer of learning rate

        for iter in range(max_iterations):
            # sample a batch of data
            xb, yb = get_batch('train', train_data, val_data)
            # evaluate the loss
            _, loss = self(xb, yb)
            losses.append(loss.item())
            
            # print the training loss
            print(f"step {iter}: train loss {loss}")
            
            # every once in a while evaluate the loss from val set
            if iter % eval_interval == 0 or iter == max_iterations - 1:
                val_x, val_y = get_batch('val', train_data, val_data)
                _, loss = self(val_x, val_y)
                print(f" val loss {loss}")
                val_losses.append(loss.item())
            
            optimizer.zero_grad(set_to_none=True) # clear previous gradients
            loss.backward() # compute gradients of all variables wrt loss
            optimizer.step() # perform updates using calculated gradients
        return losses, val_losses

def generate_model_name(name='model'): # generate a new model name
    n = 1
    while True:
        model_dir = f'LanguageModels/{name}{n}/'
        if not os.path.exists(model_dir):
            #create the directory
            os.makedirs(model_dir)
            return model_dir
        n += 1

def plot_training(losses, val_losses):
    import matplotlib.pyplot as plt
    plt.plot(losses, label='train')
    plt.figure()
    plt.plot(val_losses, label='val')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
def plot_model(model):
    # Plot the weights of the model components, with a heatmap
    import matplotlib.pyplot as plt

    # Create a figure with subplots
    _, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Plot token embedding weights
    sns.heatmap(model.token_embedding_table.weight.detach().cpu().numpy(), ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title('Token Embedding Weights')

    # Plot position embedding weights
    sns.heatmap(model.position_embedding_table.weight.detach().cpu().numpy(), ax=axes[0, 1], cmap='viridis')
    axes[0, 1].set_title('Position Embedding Weights')

    # Plot final layer norm weights
    sns.heatmap(model.ln_f.weight.detach().cpu().numpy().reshape(1, -1), ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title('Final Layer Norm Weights')

    # Plot language model head weights
    sns.heatmap(model.lm_head.weight.detach().cpu().numpy(), ax=axes[1, 1], cmap='viridis')
    axes[1, 1].set_title('Language Model Head Weights')
    
    plt.tight_layout()
    plt.show()

def main():
    print(f'Using device {device}')
#torch.manual_seed(1337)

    with open('Data/sample.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    encoding_level = input("Enter the embedding level ('word' or 'char'): ")
    stoi, itos, vocab_size = generate_mapping(text, level=encoding_level)
    
    #save the mapping
    # Check if the directory exists, if not, create it
    model_dir = generate_model_name()
    if encoding_level == 'word':
        model_dir = generate_model_name('word_model')

    # Save the mapping
    torch.save(stoi, os.path.join(model_dir, 'stoi.pth'))
    torch.save(itos, os.path.join(model_dir, 'itos.pth'))
    torch.save(vocab_size, os.path.join(model_dir, 'vocab_size.pth'))

    train_data, val_data = generate_data(text, stoi)
    
    model = TransformerLanguageModel(vocab_size)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    losses, val_losses = model.train(train_data, val_data, max_iterations)
    
    #save the losses and val_losses
    torch.save(losses, os.path.join(model_dir, 'losses.pth'))
    torch.save(val_losses, os.path.join(model_dir, 'val_losses.pth'))
    
    #save the model
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    
    plot_training(losses, val_losses)

def test():

    model_dir = 'LanguageModels/model2/'

    stoi = torch.load(os.path.join(model_dir, 'stoi.pth'), weights_only=True)
    itos = torch.load(os.path.join(model_dir, 'itos.pth'), weights_only=True)

    vocab_size = torch.load(os.path.join(model_dir, 'vocab_size.pth'), weights_only=True)

    print(vocab_size)

    model = TransformerLanguageModel(vocab_size)

    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))  # Load from current directory
    
    # Generate text
    context = 'hello what is your name'
    print('\n\n'+ Fore.YELLOW + context + Fore.RESET + '\n\n')
    
    context = torch.tensor(encode(context, stoi, level='char'), dtype=torch.long, device=device).unsqueeze(0)
    
    print(Fore.GREEN + decode(model.generate(context, max_new_tokens=500, temperature=1)[0].tolist(), itos, level='char') + Fore.RESET)
def test_word_model():

    model_dir = 'LanguageModels/constitution_word/'

    stoi = torch.load(os.path.join(model_dir, 'stoi.pth'), weights_only=True)
    itos = torch.load(os.path.join(model_dir, 'itos.pth'), weights_only=True)
    vocab_size = torch.load(os.path.join(model_dir, 'vocab_size.pth'), weights_only=True)
    
    model = TransformerLanguageModel(vocab_size)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))  # Load from current directory
    # Adjust state_dict keys to match the model's keys

    # Generate text
    context = 'hello what is your name'
    context = ' '.join(['zero'] * 10)
    print('\n\n'+ Fore.YELLOW + context + Fore.RESET + '\n\n')
    
    context = torch.tensor(encode(context, stoi, level='word'), dtype=torch.long, device=device).unsqueeze(0)
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    output = model.generate(context, max_new_tokens=500, temperature=1)[0].tolist()
    print(output)
    
    print(Fore.GREEN + decode(output, itos, level='word') + Fore.RESET)


if __name__ == '__main__':
    test_word_model()
    
