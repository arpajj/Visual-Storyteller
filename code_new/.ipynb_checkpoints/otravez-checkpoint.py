import pickle, torch
from model_t4 import T4Transformer
from t4 import T4tokenizer
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 

def plot_losses(list_of_losses, mylabel):
    plt.clf()
    iterations = range(1, len(list_of_losses) + 1)
    plt.plot(iterations, list_of_losses, color='blue', label=mylabel)
    plt.title(f'{mylabel} Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    merged_label = mylabel.replace(" ", "_")
    plt.savefig(f'./plots/phase2/auto/{merged_label}_plot.png')

name = 'admitos'
vocab_total_dii_vocab = f'/data/admitosstorage/Phase_2/data_phase2/{name.capitalize()}_way/Vocabs/{name}_dii_vocab.pkl'
vocab_total_sis_vocab = f'/data/admitosstorage/Phase_2/data_phase2/{name.capitalize()}_way/Vocabs/{name}_sis_vocab.pkl'
vocab_combined = f'/data/admitosstorage/Phase_2/data_phase2/{name.capitalize()}_way/Vocabs/all_{name}_vocab.pkl'

with open(vocab_total_dii_vocab, 'rb') as f:
    dii_vocab = pickle.load(f)

with open(vocab_total_sis_vocab, 'rb') as f:
    sis_vocab = pickle.load(f)

with open(vocab_combined, 'rb') as f:
    total_vocab = pickle.load(f)

print("The input-captions vocabulary is: ", len(dii_vocab))
print("The target-stories vocabulary is: ", len(sis_vocab))
print("The combined vocabulary is: ", len(total_vocab))

use_bert = True
if use_bert:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocabulary_size_input = len(tokenizer.vocab) # BERT's vocabulary size
    vocabulary_size_target = len(tokenizer.vocab) # BERT's vocabulary size
else:
    tokenizer = T4tokenizer(total_vocab)
    vocabulary_size_input = len(tokenizer.get_vocab())
    vocabulary_size_target = len(tokenizer.get_vocab()) 

print(vocabulary_size_input, vocabulary_size_target)
embedding_dim = 512 
number_layers = 5
number_heads = 8
feed_forward_dim = 1024
my_device = torch.device('cuda:0')
print("DEVICE USED: ", my_device)
model = T4Transformer(vocab_size_input=vocabulary_size_input, vocab_size_target=vocabulary_size_target, d_model=embedding_dim, num_layers=number_layers, 
                        num_heads=number_heads, d_ff=feed_forward_dim, dropout=0.1, pad_token=0, device=my_device)
model = model.to(my_device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
optimizer = optim.AdamW(model.parameters(), lr=0.00005)
criterion = CrossEntropyLoss(ignore_index=0)
print("The trainable parameters of the model are: ", count_parameters(model))


if use_bert: 
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
    
    with open(path_val_a, 'rb') as f:
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


if not use_bert:
    phase2_path_train_a = f'/data/admitosstorage/Phase_2/data_phase2/Combined/t4tok_train_caps.pkl'
    phase2_path_train_b = f'/data/admitosstorage/Phase_2/data_phase2/Combined/t4tok_train_stories.pkl'
    
    with open(phase2_path_train_a, 'rb') as f:
        tokenized_train_captions1 = pickle.load(f)
    
    with open(phase2_path_train_b, 'rb') as f:
        tokenized_train_stories = pickle.load(f)
    
    phase2_path_val_a = f'/data/admitosstorage/Phase_2/data_phase2/Combined/t4tok_val_caps.pkl'
    phase2_path_val_b = f'/data/admitosstorage/Phase_2/data_phase2/Combined/t4tok_val_stories.pkl'
    
    with open(phase2_path_val_a, 'rb') as f:
        tokenized_val_captions1 = pickle.load(f)
    
    with open(phase2_path_val_b, 'rb') as f:
        tokenized_val_stories = pickle.load(f)
    
    phase2_path_test_a = f'/data/admitosstorage/Phase_2/data_phase2/Combined/t4tok_test_caps.pkl'
    phase2_path_test_b = f'/data/admitosstorage/Phase_2/data_phase2/Combined/t4tok_test_stories.pkl'
    
    with open(phase2_path_test_a, 'rb') as f:
        tokenized_test_captions1 =pickle.load(f)
    
    with open(phase2_path_test_b, 'rb') as f:
        tokenized_test_stories = pickle.load(f)


## Prepareing Train Loaders 
b_s = 2
train_dataset1 = CustomDataset(tokenized_train_captions1, tokenized_train_stories)
train_loader1 = DataLoader(train_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

if use_bert:
    train_dataset2 = CustomDataset(tokenized_train_captions2, tokenized_train_stories)
    train_loader2 = DataLoader(train_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)
    
    train_dataset3 = CustomDataset(tokenized_train_captions3, tokenized_train_stories)
    train_loader3 = DataLoader(train_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

## Prepareing Validation Loaders 
val_dataset1 = CustomDataset(tokenized_val_captions1, tokenized_val_stories)
val_loader1 = DataLoader(val_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

if use_bert:
    val_dataset2 = CustomDataset(tokenized_val_captions2, tokenized_val_stories)
    val_loader2 = DataLoader(val_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)
    
    val_dataset3 = CustomDataset(tokenized_val_captions3, tokenized_val_stories)
    val_loader3 = DataLoader(val_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

## Prepareing Test Loaders 
test_dataset1 = CustomDataset(tokenized_test_captions1, tokenized_test_stories)
test_loader1 = DataLoader(test_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

if use_bert:
    test_dataset2 = CustomDataset(tokenized_test_captions2, tokenized_test_stories)
    test_loader2 = DataLoader(test_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)
    
    test_dataset3 = CustomDataset(tokenized_test_captions3, tokenized_test_stories)
    test_loader3 = DataLoader(test_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

for batch in train_loader1:
    print("Captions shape:", batch['caption_ids'][0].shape, batch['caption_ids'][1].shape)
    print("Stories shape: ", batch['story_ids'][0].shape, batch['story_ids'][0].shape)
    break


def transform_before_model(batch):
    batch_caption_ids = torch.stack(batch['caption_ids']).squeeze(1).to(my_device)
    batch_story_ids = torch.stack(batch['story_ids']).squeeze(1).to(my_device)
    return batch_caption_ids, batch_story_ids

def split_tensor(tensor, dim=1):
    size = tensor.size(dim)
    mid = size // 2
    
    # Split the tensor into two parts
    if size % 2 == 0:
        tensor1 = tensor.narrow(dim, 0, mid)
        tensor2 = tensor.narrow(dim, mid, mid)
    else:
        tensor1 = tensor.narrow(dim, 0, mid)
        tensor2 = tensor.narrow(dim, mid, mid + 1)
    
    return tensor1, tensor2


num_epochs = 4
train_loss_per_epoch = []
val_loss_per_epoch = []
start_token_id = tokenizer.cls_token_id
eos_token_id = tokenizer.sep_token_id
for epoch in range(num_epochs):
    print("------------------------------------------ Training loop ------------------------------------------------")
    epoch_train_loss = 0.0
    model.train()  
    
    train_loss_per_iter = []
    for batch_idx, batch in tqdm(enumerate(train_loader1), total=len(train_loader1),  desc="Processing Train items", ncols=100):
        src, tgt = transform_before_model(batch)
        if src.shape[1] + tgt.shape[1] > 250:
            continue
        #     print("\n Only captions bigger than 150.")
        #     caps1, caps2 = split_tensor(caps)
        #     caps_list = [caps1,caps2] 
        #     stories_list = [stories,stories]
        # elif stories.shape[1] > 150 and caps.shape[1] < 150:
        #     print("\n Only stories bigger than 150.")
        #     stor1, stor2 = split_tensor(stories)
        #     stories_list = [stor1,stor2]
        #     caps_list = [caps,caps]
        # elif caps.shape[1] > 150 and stories.shape[1] > 150:
        #     print("\n Both captions and stories bigger than 150.")
        #     caps1, caps2 = split_tensor(caps)
        #     stor1, stor2 = split_tensor(stories)
        #     caps_list = [caps1,caps2]
        #     stories_list = [stor1,stor2]
        # else:
        #     continue
        #     caps_list = [caps]
        #     stories_list = [stories]
        model.zero_grad()
        generated_sequences = torch.full((tgt.size(0), 1), start_token_id, dtype=torch.long).to(my_device)
        generation_train_loss = 0.0
        enc_mask = model.create_padding_mask(src)
        enc_out = model.encode(src, enc_mask)
        for k in range(tgt.size(1)):        
            look_ahead_mask = model.create_look_ahead_mask(generated_sequences.size(1))
            combined_mask = torch.max(model.create_padding_mask(generated_sequences), look_ahead_mask)
            dec_input = generated_sequences
            dec_out = model.decode(dec_input, enc_out, enc_mask, combined_mask)
            output = model.fc_out(dec_out)
            probs = F.softmax(output[:, -1, :], dim=-1) # put 0 in the 2nd dim for in the 3d 
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
            generated_sequences = torch.cat((generated_sequences, next_token), dim=1)
            # loss calculated per token generated
            generation_train_loss += criterion(output[:,-1,:].contiguous().view(-1,vocabulary_size_target), tgt[:, k].contiguous().view(-1)) 
            torch.cuda.empty_cache()
            #train_loss = criterion(output.contiguous().view(-1,vocabulary_size_target), stories[:, 0:k+1].squeeze(1).contiguous().view(-1)) # loss for the whole sequence
        
        generation_train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print(f"Epoch: {epoch+1}, Iteration: {batch_idx+1}/{len(train_loader1)}, Generation step: {k+1}/{stories.size(1)}, Step Loss: {train_loss.item():.5f}")
        avg_train_loss_per_iter = generation_train_loss.item()/tgt.size(1)
        train_loss_per_iter.append(avg_train_loss_per_iter)
        epoch_train_loss += avg_train_loss_per_iter
        print(f"\n Epoch: {epoch+1}, Iteration {batch_idx+1}/{len(train_loader1)}, Train Iteration Loss: {avg_train_loss_per_iter:.5f}")
        break
        
    avg_epoch_train_loss = epoch_train_loss/len(train_loader1)
    print(f"\n --------------- Epoch {epoch+1}, Average Train Epoch Loss: {avg_epoch_train_loss:.5f} ---------------------")
    train_loss_per_epoch.append(avg_epoch_train_loss)
    torch.save(model.state_dict(), f'/data/admitosstorage/Phase_2/trained_models/auto/new_model_combined_e{epoch+1}.pt')

    print("------------------------------------------ Validation loop ------------------------------------------------")
    epoch_val_loss = 0.0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for validation
        val_loss_per_iter = []
        for batch_idx, batch in tqdm(enumerate(val_loader1), total=len(val_loader1), desc="Processing Validation items", ncols=100):
            src, tgt = transform_before_model(batch)
            if src.shape[1] + tgt.shape[1] > 250:
                continue
            # if caps.shape[1] > 150 and stories.shape[1] < 150:
            #     print("\n Only captions bigger than 150.")
            #     caps1, caps2 = split_tensor(caps)
            #     caps_list = [caps1,caps2] 
            #     stories_list = [stories,stories]
            # elif stories.shape[1] > 150 and caps.shape[1] < 150:
            #     print("\n Only stories bigger than 150.")
            #     stor1, stor2 = split_tensor(stories)
            #     stories_list = [stor1,stor2]
            #     caps_list = [caps,caps]
            # elif caps.shape[1] > 150 and stories.shape[1] > 150:
            #     print("\n Both captions and stories bigger than 150.")
            #     caps1, caps2 = split_tensor(caps)
            #     stor1, stor2 = split_tensor(stories)
            #     caps_list = [caps1,caps2]
            #     stories_list = [stor1,stor2]
            # else:
            #     caps_list = [caps]
            #     stories_list = [stories]
                
            generated_sequences = torch.full((tgt.size(0), 1), start_token_id, dtype=torch.long).to(my_device)
            generation_val_loss = 0.0
            enc_mask = model.create_padding_mask(src)
            enc_out = model.encode(src, enc_mask)
            for k in range(tgt.size(1)):
                look_ahead_mask = model.create_look_ahead_mask(generated_sequences.size(1))
                combined_mask = torch.max(model.create_padding_mask(generated_sequences), look_ahead_mask)
                dec_input = generated_sequences
                dec_out = model.decode(dec_input, enc_out, enc_mask, combined_mask)
                output = model.fc_out(dec_out)             
                probs = F.softmax(output[:, -1, :], dim=-1) # put 0 in the 2nd dim for in the 3d 
                next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
                generated_sequences = torch.cat((generated_sequences, next_token), dim=1)
                generation_val_loss = criterion(output[:,-1,:].contiguous().view(-1,vocabulary_size_target), tgt[:, k].contiguous().view(-1)) # loss calculated per token generated
                torch.cuda.empty_cache()             
                #val_loss = criterion(output.contiguous().view(-1,vocabulary_size_target), stories[:, 0:k+1].squeeze(1).contiguous().view(-1)) # loss for the whole sequence
                #print(f"Epoch: {epoch+1}, Iteration: {batch_idx+1}/{len(val_loader1)}, Generation step: {k+1}/{stories.size(1)}, Step Loss: {val_loss.item():.5f}")
            
            avg_val_loss_per_iter = generation_val_loss.item()/tgt.size(1)
            val_loss_per_iter.append(avg_val_loss_per_iter)
            epoch_val_loss += avg_val_loss_per_iter
            print(f"\n Epoch: {epoch+1}, Iteration {batch_idx+1}/{len(val_loader1)}, Val Iteration Loss: {avg_val_loss_per_iter:.5f}")

        avg_epoch_val_loss = epoch_val_loss/len(val_loader1)
        print(f"\n --------------------- Epoch {epoch+1}, Average Validation Epoch Loss: {avg_epoch_val_loss:.5f} -------------------------")
        val_loss_per_epoch.append(avg_epoch_val_loss)

plot_losses(train_loss_per_iter, "Train loss")
plot_losses(val_loss_per_iter, "Validation loss")




