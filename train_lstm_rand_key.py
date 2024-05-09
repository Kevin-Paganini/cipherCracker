import torch
from torch.utils.data import DataLoader
import os
from ciphers import CryptDatasetRandKey
from sklearn.model_selection import train_test_split
from ciphers import VigenereCipher
from ciphers import CaesarCipher
from ciphers import AutokeyCipher
from models import EncoderRNN, DecoderRNN
import torch.nn as nn
import torch.optim as optim
import time
import math


vocab_char_to_num = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, ' ': 26,   '.': 27,  
    ',': 28, '?': 29, '!': 30
}

vocab_num_to_char = {}

for key in vocab_char_to_num:
    vocab_num_to_char[vocab_char_to_num[key]] = key

def string_to_numerical(s):
    return [vocab_char_to_num[c] for c in s]

def numerical_to_string(s):
    return [vocab_num_to_char[c] for c in s]



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))




def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    encoder.train()
    decoder.train()
    total_loss = 0
    
    for data in dataloader:
        input_tensor, target_tensor = data
        
        # Move data tensors to the device (GPU or CPU)
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
        
        
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        print('www', encoder_outputs.device)
        print('uuu', encoder_hidden.device)
        print('ooo', input_tensor.device)
        print('eee', target_tensor.device)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        total_loss += loss.item()
        print(f'Loss that round: {loss.item()}')
        
    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, device='', model_name=''):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        # evaluate(encoder, decoder, test_data_loader)
        torch.save(encoder.state_dict(), f'encoder_{args.model_name}.pth')
        torch.save(decoder.state_dict(), f'decoder_{args.model_name}.pth')
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))
            with open(f'log_{args.log_name}.txt', 'a') as f:
                f.write('%s (%d %d%%) %.4f\n' % (timeSince(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))


def evaluate(encoder, decoder, test_data_loader, device):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for cipher_text, clear_text in test_data_loader:
            # Move data tensors to the device (GPU or CPU)
            cipher_text, clear_text = cipher_text.to(device), clear_text.to(device)
            
            encoder_outputs, encoder_hidden = encoder(cipher_text)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
            
            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            
            decoded_letters = []
            decoded_messages = [numerical_to_string(x.tolist()) for x in decoded_ids]
            clear_text = [[numerical_to_string(x.tolist()) for x in clear_text]]
            
        print(clear_text)
        print(decoded_messages)


import argparse

parser = argparse.ArgumentParser(prog="Train NN on task")

parser.add_argument('--epochs', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--cipher')
parser.add_argument('--key')
parser.add_argument('--model_name')
parser.add_argument('--log_name')
args = parser.parse_args()

print(args)

# Set the device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Instantiate ciphers based on args
if args.cipher == 'A':
    train_data = CryptDatasetRandKey(AutokeyCipher(args.key), os.path.join('data', 'amazon_train.txt'))
    test_data = CryptDatasetRandKey(AutokeyCipher(args.key), os.path.join('data', 'amazon_test.txt'))
elif args.cipher == 'C':
    train_data = CryptDatasetRandKey(CaesarCipher(4), os.path.join('data', 'amazon_train.txt'))
    test_data = CryptDatasetRandKey(CaesarCipher(4), os.path.join('data', 'amazon_test.txt'))
elif args.cipher == 'V':
    train_data = CryptDatasetRandKey(VigenereCipher(args.key), os.path.join('data', 'amazon_train.txt'))
    test_data = CryptDatasetRandKey(VigenereCipher(args.key), os.path.join('data', 'amazon_test.txt'))
else:
    raise NotImplementedError("Cipher not implemented")

# Create data loaders
train_data_loader = DataLoader(train_data, batch_size=2048, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=512, shuffle=True)




# Define models
encoder = EncoderRNN(31, args.hidden_size).to(device)
decoder = DecoderRNN(args.hidden_size, 31).to(device)

# Train models
train(train_data_loader, encoder, decoder, args.epochs, print_every=1, device=device)

# Save trained models
torch.save(encoder.state_dict(), f'encoder_{args.model_name}.pth')
torch.save(decoder.state_dict(), f'decoder_{args.model_name}.pth')
