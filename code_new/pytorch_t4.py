import torch, random
import torch.nn as nn
import torch.optim as optim
import math, pickle
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from transformers import BertTokenizer

path_train_a = f'/data/admitosstorage/Phase_2/data_phase2/Combined/train_caps_1.pkl'
path_train_b = f'/data/admitosstorage/Phase_2/data_phase2/Combined/train_caps_2.pkl'
path_train_c = f'/data/admitosstorage/Phase_2/data_phase2/Combined/train_caps_3.pkl'

with open(path_train_a, 'rb') as f:
    tokenized_train_captions1 = pickle.load(f)

with open(path_train_b, 'rb') as f:
    tokenized_train_captions2 = pickle.load(f)

with open(path_train_c, 'rb') as f:
    tokenized_train_captions3 = pickle.load(f)

path_val_a = f'/data/admitosstorage/Phase_2/data_phase2/Combined/val_caps_1.pkl'
path_val_b = f'/data/admitosstorage/Phase_2/data_phase2/Combined/val_caps_2.pkl'
path_val_c = f'/data/admitosstorage/Phase_2/data_phase2/Combined/val_caps_3.pkl'

with open(path_val_a, 'rb') as f:
    tokenized_val_captions1 = pickle.load(f)

with open(path_val_b, 'rb') as f:
    tokenized_val_captions2 = pickle.load(f)

with open(path_val_c, 'rb') as f:
    tokenized_val_captions3 = pickle.load(f)
    
path_test_a = f'/data/admitosstorage/Phase_2/data_phase2/Combined/test_caps_1.pkl'
path_test_b = f'/data/admitosstorage/Phase_2/data_phase2/Combined/test_caps_2.pkl'
path_test_c = f'/data/admitosstorage/Phase_2/data_phase2/Combined/test_caps_3.pkl'

with open(path_test_a, 'rb') as f:
    tokenized_test_captions1 = pickle.load(f)

with open(path_test_b, 'rb') as f:
    tokenized_test_captions2 = pickle.load(f)

with open(path_test_c, 'rb') as f:
    tokenized_test_captions3 = pickle.load(f)

path_train_stories = f'/data/admitosstorage/Phase_2/data_phase2/Combined/train_stories.pkl'
path_val_stories = f'/data/admitosstorage/Phase_2/data_phase2/Combined/val_stories.pkl'
path_test_stories = f'/data/admitosstorage/Phase_2/data_phase2/Combined/test_stories.pkl'

with open(path_train_stories, 'rb') as f:
    tokenized_train_stories = pickle.load(f)

with open(path_val_stories, 'rb') as f:
    tokenized_val_stories = pickle.load(f)

with open(path_test_stories, 'rb') as f:
    tokenized_test_stories = pickle.load(f)

class CustomDataset(Dataset):
    def __init__(self, entries, references):
        self.entries = entries
        self.references = references

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = {}
        item['caption'] = self.entries[idx]
        item['story'] = self.references [idx] 
        return item

        
def my_collate_fn(batch):
    input_ids = [item['caption'].transpose(0,1) for item in batch]
    target_ids = [item['story'].transpose(0,1) for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)

    final_input_ids = [item.transpose(0,1) for item in input_ids_padded]
    final_labels = [item.transpose(0,1) for item in target_ids_padded]

    return {'caption_ids': final_input_ids, 'story_ids': final_labels}

## Prepareing Train Loaders 
b_s = 10
train_dataset1 = CustomDataset(tokenized_train_captions1, tokenized_train_stories)
train_loader1 = DataLoader(train_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)
print(len(train_loader1))

train_dataset2 = CustomDataset(tokenized_train_captions2, tokenized_train_stories)
train_loader2 = DataLoader(train_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

train_dataset3 = CustomDataset(tokenized_train_captions3, tokenized_train_stories)
train_loader3 = DataLoader(train_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

## Prepareing Validation Loaders 
val_dataset1 = CustomDataset(tokenized_val_captions1, tokenized_val_stories)
val_loader1 = DataLoader(val_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)
print(len(val_loader1))

val_dataset2 = CustomDataset(tokenized_val_captions2, tokenized_val_stories)
val_loader2 = DataLoader(val_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

val_dataset3 = CustomDataset(tokenized_val_captions3, tokenized_val_stories)
val_loader3 = DataLoader(val_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

## Prepareing Test Loaders 
test_dataset1 = CustomDataset(tokenized_test_captions1, tokenized_test_stories)
test_loader1 = DataLoader(test_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)
print(len(test_loader1))

test_dataset2 = CustomDataset(tokenized_test_captions2, tokenized_test_stories)
test_loader2 = DataLoader(test_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

test_dataset3 = CustomDataset(tokenized_test_captions3, tokenized_test_stories)
test_loader3 = DataLoader(test_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

### Function for applying Mask Language Modelling
def mask_tokens(labels, tokenizer, mlm_probability=0.15):
    inputs = labels.clone()
    labels = labels.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs.to(device), labels.to(device)

### Functions for applying Sentence Permutation
def pad_to_same_size(tensor1, tensor2):
    # Get the shapes of the two tensors
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    
    # Determine the maximum size along each dimension
    max_rows = max(shape1[0], shape2[0])
    max_cols = max(shape1[1], shape2[1])
    
    # Create new tensors with the maximum size, initialized with zeros
    padded_tensor1 = torch.zeros((max_rows, max_cols), dtype=tensor1.dtype)
    padded_tensor2 = torch.zeros((max_rows, max_cols), dtype=tensor2.dtype)
    
    # Copy the values from the original tensors to the new tensors
    padded_tensor1[:shape1[0], :shape1[1]] = tensor1
    padded_tensor2[:shape2[0], :shape2[1]] = tensor2
    
    return padded_tensor1.to(device), padded_tensor2.to(device)

def permute_sentences(stories, tokenizer):
    permuted_stories = []
    for story in stories:
        # Decode the story to get the text
        story_text = tokenizer.decode(story, skip_special_tokens=True)
        #print(len(story_text.split()))
        # Split the story into sentences
        sentences = story_text.split('.')
        # Remove empty strings
        sentences = [s for s in sentences if s]
        # Shuffle the sentences
        random.shuffle(sentences)
        # Join the sentences back into a single string
        permuted_story = '. '.join(sentences) + '.'
        # Encode the permuted story back to tokens
        permuted_stories.append(tokenizer.encode(permuted_story, add_special_tokens=True))
    
    # Pad the permuted stories to the same length
    max_length = max(len(s) for s in permuted_stories)
    permuted_stories = [s + [tokenizer.pad_token_id] * (max_length - len(s)) for s in permuted_stories]

    return torch.tensor(permuted_stories).to(device)


##################  ------------------------------  Setting the model ---------------------------- ######################
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
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
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
        transformer_out = self.transformer(src, tgt, src_mask=None, tgt_mask=tgt_mask)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
## Parametrization of the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Transformer(num_tokens=30522, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1, pad_token=0).to(device)
opt = optim.AdamW(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
num_epochs = 15
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=10000, num_training_steps=int(num_epochs)*len(train_loader1))
print("The trainable parameters of the model are: ", count_parameters(model))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load(f'/data/admitosstorage/Phase_2/trained_models/united_model_pytorch_mlm_sp_e30.pt', map_location=device))
model = model.to(device)
#model.load_state_dict(torch.load(f'/data/admitosstorage/Phase_2/trained_models/model_pytorch_e10_v2.pt', map_location=device)) # Maybe from epoch 4 would be better

def transform_before_model(batch):
    batch_caption_ids = torch.stack(batch['caption_ids']).squeeze(1).to(device)
    batch_story_ids = torch.stack(batch['story_ids']).squeeze(1).to(device)
    return batch_caption_ids, batch_story_ids

for batch in train_loader1:
    print("Captions shape:", batch['caption_ids'][0].shape, batch['caption_ids'][1].shape)
    print("Stories shape: ", batch['story_ids'][0].shape, batch['story_ids'][0].shape)
    break

def train_loop(model, opt, loss_fn, dataloader, epoch, type_train):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    print(" ############################################## Training Loop ############################################### ")
    model.train()
    total_loss = 0
    for idx,batch in enumerate(dataloader):
        opt.zero_grad()
        caps, stories = transform_before_model(batch)
        if type_train.upper() == 'MLM':
            masked_dec_inputs, masked_labels = mask_tokens(stories.to('cpu'), tokenizer)
            masked_y_inputs = masked_dec_inputs[:,:-1]
            masked_y_expected = masked_labels[:,1:]

            src_mask = model.get_src_tgt_mask(caps.size(1)).to(device)
            sequence_length = masked_y_inputs.size(1)
            tgt_mask = model.get_src_tgt_mask(sequence_length)
            
            pred = model(caps, masked_y_inputs, src_mask, tgt_mask)
            pred = pred.permute(0, 2, 1) 
            loss = loss_fn(pred, masked_y_expected)           
            
        elif type_train.upper() == 'SP':
            permuted_stories = permute_sentences(stories.to('cpu'), tokenizer)
            if not stories.shape == permuted_stories.shape:
                stories, permuted_stories = pad_to_same_size(stories, permuted_stories)
            y_input_permuted = permuted_stories[:,:-1]   
            y_expected = stories[:,1:]
            
            src_mask = model.get_src_tgt_mask(caps.size(1)).to(device)
            sequence_length = y_input_permuted.size(1)
            tgt_mask = model.get_src_tgt_mask(sequence_length)
            
            pred = model(caps, y_input_permuted, src_mask, tgt_mask)
            pred = pred.permute(0, 2, 1) 
            loss = loss_fn(pred, y_expected) 
            
        else:
            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = stories[:,:-1]
            y_expected = stories[:,1:]

            # Get the mask of the input sequence
            src_mask = model.get_src_tgt_mask(caps.size(1)).to(device)
            
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_src_tgt_mask(sequence_length)
    
            # Standard training except we pass in y_input and tgt_mask
            pred = model(caps, y_input, src_mask, tgt_mask)
    
            # Permute pred to have batch size first again
            pred = pred.permute(0, 2, 1) 
            loss = loss_fn(pred, y_expected)
            
        print(f'Epoch: {epoch}, Iteration: {idx}/{len(dataloader)}, with running loss: {loss.item()}')   
        loss.backward()
        opt.step()
        scheduler.step()     
        total_loss += loss.detach().item()
 
    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader, epoch, type_valid):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    model.eval()
    total_loss = 0
    print(" ############################################## Validation Loop ############################################### ")
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            caps, stories = transform_before_model(batch)
            if type_valid.upper() == 'MLM':
                masked_dec_inputs, masked_labels = mask_tokens(stories.to('cpu'), tokenizer)
                masked_y_inputs = masked_dec_inputs[:,:-1]
                masked_y_expected = masked_labels[:,1:]
    
                src_mask = model.get_src_tgt_mask(caps.size(1)).to(device)
                sequence_length = masked_y_inputs.size(1)
                tgt_mask = model.get_src_tgt_mask(sequence_length)
                
                pred = model(caps, masked_y_inputs, src_mask, tgt_mask)
                pred = pred.permute(0, 2, 1) 
                loss = loss_fn(pred, masked_y_expected)
                
            elif type_valid.upper() == 'SP':
                permuted_stories = permute_sentences(stories.to('cpu'), tokenizer)
                if not stories.shape == permuted_stories.shape:
                    stories, permuted_stories = pad_to_same_size(stories, permuted_stories)
                y_input_permuted = permuted_stories[:,:-1]   
                y_expected = stories[:,1:]
                
                src_mask = model.get_src_tgt_mask(caps.size(1)).to(device)
                sequence_length = y_input_permuted.size(1)
                tgt_mask = model.get_src_tgt_mask(sequence_length)
                
                pred = model(caps, y_input_permuted, src_mask, tgt_mask)
                pred = pred.permute(0, 2, 1) 
                loss = loss_fn(pred, y_expected) 
            else:
                # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
                y_input = stories[:,:-1]
                y_expected = stories[:,1:]
                
                # Get mask to mask out the next words
                src_mask = model.get_src_tgt_mask(caps.size(1)).to(device)
                sequence_length = y_input.size(1)
                tgt_mask = model.get_src_tgt_mask(sequence_length)
    
                # Standard training except we pass in y_input and src_mask
                pred = model(caps, y_input, src_mask, tgt_mask)
                #pred = model(caps, y_input)
                
                # Permute pred to have batch size first again
                pred = pred.permute(0, 2, 1)     
                loss = loss_fn(pred, y_expected)
            
            print(f'Epoch: {epoch}, Iteration: {idx}/{len(dataloader)}, with running loss: {loss.item()}')
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)
    

def fit(model, opt, loss_fn, train_dataloaders, val_dataloaders, epochs, type_of_train):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        if epoch <= int(epochs//3) - 1:
            train_dataloader, val_dataloader = train_dataloaders[0], val_dataloaders[0]
        elif epoch >= int(epochs//3) and epoch < 2*int(epochs//3):
            train_dataloader, val_dataloader = train_dataloaders[1], val_dataloaders[1]
        else:
            train_dataloader, val_dataloader = train_dataloaders[2], val_dataloaders[2]
            
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, epoch, type_of_train)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader, epoch, type_of_train)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
        if (epoch+1)%3==0:
            torch.save(model.state_dict(), f'/data/admitosstorage/Phase_2/trained_models/final_model_pytorch_all_e{epoch+1}.pt')
    return train_loss_list, validation_loss_list

xx = input("Train? [y/n]")
if xx == 'y':
    pass
else:
    exit()

yy = input("How to train? [MLM/SP/other]: ")
train_loss_list, validation_loss_list = fit(model, opt, loss_fn, [train_loader1,train_loader2,train_loader3], 
                                            [val_loader1,val_loader2,val_loader3], num_epochs, yy)

def plot_losses(train_losses, validation_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, color='blue', label='Train Loss')
    plt.plot(epochs, validation_losses, color='orange', label='Validation Loss')
    plt.title('Train/Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./plots/phase2/losses_pytorch_t4_final_all.png')


plot_losses(train_loss_list,validation_loss_list)