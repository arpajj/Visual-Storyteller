import torch, re
import torch.nn as nn
import torch.optim as optim
import math, random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import json, random, pickle
import torch.nn.functional as F


train_path1 = '/data/admitosstorage/Phase_2/data_phase2/new_train_dataset1.json'
val_path1 = '/data/admitosstorage/Phase_2/data_phase2/new_val_dataset1.json'
test_path1 = '/data/admitosstorage/Phase_2/data_phase2/new_test_dataset1.json'

with open(train_path1, 'r', encoding='utf-8') as f:
    my_new_train_dataset = dict(list(json.load(f).items())[:-1]) # We dont want to load the title 

with open(val_path1, 'r', encoding='utf-8') as f:
    my_new_val_dataset = dict(list(json.load(f).items())[:-1]) # We dont want to load the title 

with open(test_path1, 'r', encoding='utf-8') as f:
    my_new_test_dataset = dict(list(json.load(f).items())[:-1]) # We dont want to load the title  


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float()*(-math.log(10000.0))/dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(self,num_tokens,dim_model,num_heads,num_encoder_layers,num_decoder_layers,dropout_p,pad_token):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.pad_token = pad_token
        
        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model,nhead=num_heads,num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src)*math.sqrt(self.dim_model)
        tgt = self.embedding(tgt)*math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
     
        # we permute to obtain size (sequence length, batch_size, dim_model),
        # src = src.permute(1, 0, 2)
        # tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence_length, batch_size, num_tokens)
        # With batch_first = True it is: (batch_size, sequence_length, num_tokens)
        transformer_out = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.out(transformer_out)

        return out

    def get_src_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token_id = tokenizer.cls_token_id
eos_token_id = tokenizer.sep_token_id


def beam_search(model, input_sequence, SOS_token, EOS_token, beam_width, max_length=149):
    device = next(model.parameters()).device  # Get device from model parameters

    # Initialize sequences with the <cls> token (or <s> token)
    y_input = torch.tensor([[SOS_token]], dtype=torch.long).to(device)
    gen_seq = torch.tensor([[SOS_token]], dtype=torch.long).to(device)

    # Initialize beams with initial sequence
    beams = [(gen_seq, 0)]  # List of (sequence, cumulative log probability)

    src_mask = model.get_src_tgt_mask(input_sequence.size(1))

    for k in range(max_length):
        new_beams = []
        for seq, score in beams:
            y_input = seq.to(device)
            tgt_mask = model.get_src_tgt_mask(y_input.size(1)).to(device)
            pred = model(input_sequence, y_input, src_mask, tgt_mask)

            # Get top beam_width predictions for each sequence
            topk = torch.topk(pred[:, -1, :], beam_width, dim=-1)
            topk_indices = topk.indices.squeeze(0)
            topk_scores = topk.values.squeeze(0)

            for i in range(beam_width):
                next_token = topk_indices[i].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]
                next_score = topk_scores[i].item()

                new_seq = torch.cat([seq, next_token], dim=-1)  # Append next token to sequence
                new_beams.append((new_seq, score + next_score))

        # Sort beams by score and keep top beam_width sequences
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Update beams with new sequences
        beams = new_beams

        # Check if all beams have ended with EOS_token
        all_ended = all(b[0][0, -1].item() == EOS_token for b in beams)
        if all_ended:
            break

    # Select the sequence with the highest score as the final generated sequence
    final_seq = max(beams, key=lambda x: x[1])[0]
    return final_seq.view(-1).tolist()


def predict(model, input_sequence, target_sequence, SOS_token, EOS_token, max_length=149, teacher_forcing=False, strategy='greedy', p=0.9):
    model.eval()
    
    def capitalize_story(input_story):
        # Capitalize the first letter of the entire story
        input_story = input_story.strip().capitalize()
        # Use regex to find all occurrences of a full stop followed by a space and a letter, and capitalize that letter
        input_story = re.sub(r'(\. )([a-z])', lambda match: match.group(1) + match.group(2).upper(), input_story)
        return input_story
        
    if strategy == 'beam_search':
        #beam_width = input("Please choose the beam width (Int from 1 to 5): ")
        beam_width = 3
        generated_sequence = beam_search(model, input_sequence, SOS_token, EOS_token, int(beam_width))
        story = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        final_story = capitalize_story(story)  
        return generated_sequence, final_story
    
    y_input = torch.tensor([[SOS_token]], dtype=torch.long).to(device)
    gen_seq = torch.tensor([[SOS_token]], dtype=torch.long).to(device)
    src_mask = model.get_src_tgt_mask(input_sequence.size(1))
    
    def nucleus_sampling(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        return torch.multinomial(F.softmax(logits, dim=-1), 1).item()

    for k in range(max_length):
        tgt_mask = model.get_src_tgt_mask(y_input.size(1)).to(device)
        pred = model(input_sequence, y_input, src_mask, tgt_mask)
        
        if strategy == 'greedy':
            next_item = pred.topk(1)[1].view(-1)[-1].item()
        elif strategy == 'multinomial':
            next_item = torch.multinomial(F.softmax(pred[:, -1, :], dim=-1), 1).item()
        elif strategy == 'p-sampling':
            next_item = nucleus_sampling(pred[:, -1, :], p)
        else:
            raise ValueError("Unsupported decoding strategy")
        
        next_item = torch.tensor([[next_item]]).to(device)
        gen_seq = torch.cat((y_input, next_item), dim=1)
        
        if teacher_forcing and k % 2 == 0:
            y_input = torch.cat((y_input, target_sequence[:, k:k+1]), dim=1)
        else:
            y_input = gen_seq
        
        if next_item.view(-1).item() == EOS_token:
            break

    story_token_ids = gen_seq.view(-1).tolist()
    story = tokenizer.decode(story_token_ids, skip_special_tokens=True)
        
    final_story = capitalize_story(story)
    return story_token_ids, final_story


device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = Transformer(num_tokens=30522, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1, pad_token=0).to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print("The trainable parameters of the model are: ", count_parameters(model))
model.load_state_dict(torch.load(f'/data/admitosstorage/Phase_2/trained_models/final_model_pytorch_all_e12.pt', map_location=device))


# rand_choice = random.randint(0,len(my_new_test_dataset))
# rand_choice = 218
# new_entry = list(my_new_train_dataset.items())[rand_choice][1]
# new_captions = new_entry[0]
# new_story = new_entry[1]
# input_text = ' [SEP] '.join(new_captions)
# print(input_text)
# print()
# print(new_story)
# caption_ids = tokenizer(input_text, return_tensors="pt").input_ids
# target_ids = tokenizer(new_story, return_tensors="pt").input_ids
# print()

# decoding_strategy = input("Please choose one of the following decoding strategies [greedy/multinomial/p-sampling/beam_search]: ")
# generated_story_ids, my_story = predict(model, caption_ids.to(device), target_ids.to(device), start_token_id, eos_token_id, teacher_forcing=False, strategy=decoding_strategy)
# print(my_story)

def T4_testing_function(tested_model, test_dataset, tokenizer, my_strategy, device):
    results_stories = []
    for i in range(len(test_dataset)):
        if i%2 == 0:
            print(f"On story generation {i} out of {len(my_new_test_dataset)}")
        new_entry = list(test_dataset.items())[i][1]
        new_captions = new_entry[0]
        new_story = new_entry[1]
        input_text = ' [SEP] '.join(new_captions)
        caption_ids = tokenizer(input_text, return_tensors="pt").input_ids
        target_ids = tokenizer(new_story, return_tensors="pt").input_ids
        generated_story_ids, my_story = predict(tested_model, caption_ids.to(device), target_ids.to(device), start_token_id, eos_token_id, teacher_forcing=False, strategy=my_strategy)

        results_stories.append(my_story)
    return results_stories

t4_mlmsp_beam_stories = T4_testing_function(model, my_new_test_dataset, tokenizer, 'beam_search', device)
path_res = f'/data/admitosstorage/Phase_2/results/t4-results/res_t4_mlmsp_beam.pkl' 

with open(path_res, 'wb') as f:
    pickle.dump(t4_mlmsp_beam_stories, f)
