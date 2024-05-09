from torch.utils.data import Dataset
import torch
import random
from abc import ABC, abstractmethod




LETTER_OFFSET = 65
# A will be equal to 0, B = 1, ...
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



class Cipher(ABC):
    @abstractmethod
    def cipher_text(self, text):
        """Encrypts the given text and returns the ciphered text."""
        pass
    
    @abstractmethod
    def generate_key(self):
        """Generates a new key for the cipher and updates the current key."""
        pass
    
    @abstractmethod
    def original_text(self, cipher_text):
        """Decrypts the given ciphered text and returns the original text."""
        pass



class VigenereCipher(Cipher):
    def __init__(self, key):
        self.key = key
        

    def cipher_text(self, text):
        cipher_text = ""
        key_index = 0
        text = text.upper()
        for char in text:
            x = (vocab_char_to_num[char] + vocab_char_to_num[self.key[key_index]]) % len(vocab_char_to_num)

            cipher_text += vocab_num_to_char[x]
            key_index = (key_index + 1) % len(self.key)
        return cipher_text
    
    def generate_key(self):
        key_len = random.randint(5, 15)
        key = ''
        for i in range(key_len):
            key += vocab_num_to_char[random.randint(0, 30)]
        self.key = key
        print(f'key is now: {self.key}')

        return self

    def original_text(self, cipher_text):
        orig_text = ""
        key_index = 0
        for char in cipher_text:
            x = (vocab_char_to_num[char] - vocab_char_to_num[self.key[key_index]]) % len(vocab_char_to_num)
            
            orig_text += vocab_num_to_char[x]
            key_index = (key_index + 1) % len(self.key)
        return orig_text

    

class CaesarCipher(Cipher):
    def __init__(self, shift):
        self.shift = shift

    def cipher_text(self, text):
        cipher_text = ""
        text = text.upper()
        for char in text:
            x = (vocab_char_to_num[char] + self.shift) % len(vocab_char_to_num)
            
            cipher_text += vocab_num_to_char[x]
        return cipher_text
    
    def generate_key(self):
        self.shift = random.randint(0, 30)
        print(f'key is now: {self.shift}')
        return self

    def original_text(self, cipher_text):
        orig_text = ""
        for char in cipher_text:
            x = (vocab_char_to_num[char] - self.shift) % len(vocab_char_to_num)
            orig_text += vocab_num_to_char[x]
        return orig_text


class AutokeyCipher(Cipher):
    def __init__(self, key):
        # The initial key provided
        self.key = key.upper()

    def cipher_text(self, text):
        text = text.upper()
        # Generate the full key by appending the plaintext to the key
        full_key = self.key + text[:len(text) - len(self.key)]
        cipher_text = ""
        # Perform the encryption
        for i, char in enumerate(text):
            # Calculate the shift
            shift = vocab_char_to_num[full_key[i]]
            # Apply the shift to the character and get the encrypted character
            x = (vocab_char_to_num[char] + shift) % len(vocab_char_to_num)
            cipher_text += vocab_num_to_char[x]
        return cipher_text
    
    
    def generate_key(self):
        key_len = random.randint(5, 15)
        key = ''
        for i in range(key_len):
            key += vocab_num_to_char[random.randint(0, 30)]
            
        self.key = key
        print(f'key is now: {self.key}')
        
        return self
        

    def original_text(self, cipher_text):
        # Generate the full key by using the provided key
        full_key = self.key
        orig_text = ""
        for i, char in enumerate(cipher_text):
            # Calculate the shift
            shift = vocab_char_to_num[full_key[i]]
            # Apply the reverse shift to the character and get the decrypted character
            x = (vocab_char_to_num[char] - shift) % len(vocab_char_to_num)
            orig_text += vocab_num_to_char[x]
            # Add the decrypted character to the full key to expand the key
            if len(full_key) < len(cipher_text):
                full_key += orig_text[-1]
        return orig_text


class CryptDataset(Dataset):
    def __init__(self, cipher, clear_text_file):
        with open(clear_text_file, 'r') as f:
            clear_text_samples = f.read().split('\n')
        
        self.clear_text_samples = [x.strip() for x in clear_text_samples]
        self.padded_clear_text_samples = self._pad(clear_text_samples)
        self.padded_cipher_text_samples = [cipher.cipher_text(x) for x in self.padded_clear_text_samples]
        self.encoded_clear_text_samples = torch.tensor([string_to_numerical(x) for x in self.padded_clear_text_samples])
        self.encoded_cipher_text_samples = torch.tensor([string_to_numerical(x) for x in self.padded_cipher_text_samples])
        
        

    def _pad(self, samples):
        max_len = max([len(x) for x in samples])
        return_samples = []
        for t in samples:
            if len(t) < max_len:
                t += ' ' * (max_len - len(t))
            return_samples.append(t)
        self.max_len = max_len
        return return_samples
    
    def __len__(self):
        return len(self.clear_text_samples)
    
    def __getitem__(self, idx):
        return self.encoded_cipher_text_samples[idx], self.encoded_clear_text_samples[idx]

                         




class CryptDatasetRandKey(Dataset):
    def __init__(self, cipher, clear_text_file):
        with open(clear_text_file, 'r') as f:
            clear_text_samples = f.read().split('\n')
        
        self.clear_text_samples = [x.strip() for x in clear_text_samples]
        self.padded_clear_text_samples = self._pad(clear_text_samples)
        
        
        self.padded_cipher_text_samples = [cipher.generate_key().cipher_text(x) for x in self.padded_clear_text_samples]
        self.encoded_clear_text_samples = torch.tensor([string_to_numerical(x) for x in self.padded_clear_text_samples])
        self.encoded_cipher_text_samples = torch.tensor([string_to_numerical(x) for x in self.padded_cipher_text_samples])
        
        

    def _pad(self, samples):
        max_len = max([len(x) for x in samples])
        return_samples = []
        for t in samples:
            if len(t) < max_len:
                t += ' ' * (max_len - len(t))
            return_samples.append(t)
        self.max_len = max_len
        return return_samples
    
    def __len__(self):
        return len(self.clear_text_samples)
    
    def __getitem__(self, idx):
        return self.encoded_cipher_text_samples[idx], self.encoded_clear_text_samples[idx]

