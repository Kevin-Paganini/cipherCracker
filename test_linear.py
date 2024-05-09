from ciphers import VigenereCipher
from ciphers import CaesarCipher
from models import LinearCracker
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np


def string_to_numerical(s):
    return [ord(c) - ord('A') + 1 for c in s]

def numerical_to_string(nums):
    try:
        val = ''.join(chr(n + ord('A') - 1) for n in nums)
    except ValueError as e:
        print(e)
        val = 'A' * len(nums)
    return val

def round_list(l):
    return [round(x) for x in l]


with open('data\\words.txt', encoding='utf-8') as f:
    content = f.read()
    content = content.upper()
    
five_letter_words = content.split('\n')
words = five_letter_words

model = LinearCracker(1000, 5)

v = VigenereCipher("AAAAA")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

batch_size = 1024
num_batches = len(words) // batch_size

for i in range(10000):
    for j in range(num_batches):
        batch_words = words[j * batch_size:(j + 1) * batch_size]
        clear_text_nums = [string_to_numerical(word) for word in batch_words]
        cipher_text_nums = [string_to_numerical(v.cipher_text(word)) for word in batch_words]

        clear_text_nums = torch.tensor(clear_text_nums, dtype=torch.float32)
        cipher_text_nums = torch.tensor(cipher_text_nums, dtype=torch.float32)

        optimizer.zero_grad()

        predicted_clear_text_nums = model(cipher_text_nums)
        loss = criterion(predicted_clear_text_nums, clear_text_nums)

        if j % 10 == 0:
            print(f'Batch {j+1} of {num_batches}')
            print(f'Loss: {loss}')

        loss.backward()
        optimizer.step()