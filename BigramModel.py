import os
import random

def train_bigram_model(text): 
    bigrams = {}
    for i in range(len(text) - 1):
        curr_char = text[i]
        next_char = text[i + 1]
        if curr_char not in bigrams:
            bigrams[curr_char] = {}
        if next_char not in bigrams[curr_char]:
            bigrams[curr_char][next_char] = 0
        bigrams[curr_char][next_char] += 1

    # Convert counts to probabilities
    for curr_char, next_chars in bigrams.items():
        total = sum(next_chars.values())
        for next_char in next_chars:
            next_chars[next_char] /= total

    return bigrams

def generate_text(model, length=100):
    curr_char = random.choice(list(model.keys()))
    result = [curr_char]
    for _ in range(length - 1):
        next_chars = model.get(curr_char)
        if not next_chars:
            curr_char = random.choice(list(model.keys()))
            result.append(curr_char)
            continue
        next_char = random.choices(
            list(next_chars.keys()), weights=list(next_chars.values())
        )[0]
        result.append(next_char)
        curr_char = next_char
    return ''.join(result)

def main():
    data_path = 'Data/sample.txt'
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    model = train_bigram_model(text)
    generated_text = generate_text(model, length=500)
    print(generated_text)

if __name__ == '__main__':
    main()