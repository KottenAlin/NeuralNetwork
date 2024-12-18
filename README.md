# Project Title

This project, developed by me Sebastian as a school AI project, aims to understand AI models and language models. The project involves building several models, starting with a simple Multi-Layer Perceptron (MLP) neural network, followed by a bigram model, an autoencoder (which did not work), and finally a transformer model. The transformer model is designed to take a text as training data and output text similar to the input text.

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Functionality](#functionality)
  - [Results](#results)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Installation


```bash
# Clone the repository
git clone https://github.com/KottenAlin/NeuralNetwork.git
# Navigate to the project directory

cd NeuralNetwork
# Install dependencies
pip install -r requirements.txt

```

## Functionality

This project explores AI models by building several neural networks from scratch using Python and PyTorch. The models developed include:

- **Simple Neural Network (MLP)**: This model implements a basic multi-layer perceptron (MLP), with numpy for very simple training data.
- **Bigram Model**: This model creates a simple bigram language model that predicts the next character in a sequence based on the previous character.
- **Autoencoder**: Although this model did not work as expected, it was an attempt to learn data representations in an unsupervised manner. An autoencoder consists of an encoder that compresses the input into a latent-space representation and a decoder that reconstructs the input from this representation.
- **Transformer Model**: This advanced model develops a transformer-based language model. It features embedding layers to convert words into vectors, multi-head self-attention mechanisms to capture relationships between words in a sequence, feed-forward networks for processing these relationships, and positional encoding to retain the order of words/charcters. The transformer model is trained on textual data to generate text that resembles the training data.

Additionally, the project includes an interface for interacting with the trained models. This interface allows users to:

- **Chat with the AI**: Engage in conversations with the AI to see how it generates responses based on the trained models.
- **Visualize Model Components**: View and understand the different components and layers of the models, aiding in the learning process.
- **Train New Models**: Use the provided tools and scripts to train new models on different datasets, facilitating experimentation and further learning.

## Results

The results of the simple neural network (MLP) were satisfactory for very basic training data, but it struggled with more complex data. To delve deeper into language processing, I developed a bigram model, which performed relatively well. However, since it only predicts the next character based on the previous one, its performance was limited. 

Next, I attempted to create an autoencoder to learn data representations in an unsupervised manner, but unfortunately, this model did not work as expected. 

Finally, I built a transformer model, which is well-suited for sequential data like text. This model performed quite well, but due to its small size (larger models would require significantly more time to train), it struggled with capturing context effectively. I also experimented with word-level encoding instead of character-level encoding, but this approach did not yield satisfactory results.

## Usage

Instructions on how to use the project.

## Contributing

This is currently not being developed, so further conttribution is not

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.