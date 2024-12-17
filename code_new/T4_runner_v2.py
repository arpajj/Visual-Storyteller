from t4 import T4Transformer
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from torch.nn import CrossEntropyLoss
import pickle, torch, os, random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import torch.nn.functional as F
#from torch.nn import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


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
b_s = 1
train_dataset1 = CustomDataset(tokenized_train_captions1, tokenized_train_stories)
train_loader1 = DataLoader(train_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

train_dataset2 = CustomDataset(tokenized_train_captions2, tokenized_train_stories)
train_loader2 = DataLoader(train_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

train_dataset3 = CustomDataset(tokenized_train_captions3, tokenized_train_stories)
train_loader3 = DataLoader(train_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

## Prepareing Validation Loaders 
val_dataset1 = CustomDataset(tokenized_val_captions1, tokenized_val_stories)
val_loader1 = DataLoader(val_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

val_dataset2 = CustomDataset(tokenized_val_captions2, tokenized_val_stories)
val_loader2 = DataLoader(val_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

val_dataset3 = CustomDataset(tokenized_val_captions3, tokenized_val_stories)
val_loader3 = DataLoader(val_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

## Prepareing Test Loaders 
test_dataset1 = CustomDataset(tokenized_test_captions1, tokenized_test_stories)
test_loader1 = DataLoader(test_dataset1, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

test_dataset2 = CustomDataset(tokenized_test_captions2, tokenized_test_stories)
test_loader2 = DataLoader(test_dataset2, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

test_dataset3 = CustomDataset(tokenized_test_captions3, tokenized_test_stories)
test_loader3 = DataLoader(test_dataset3, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

for batch in train_loader1:
    print("Captions shape:", batch['caption_ids'][0].shape)
    print("Stories shape: ", batch['story_ids'][0].shape)
    break

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocabulary_size_input = len(tokenizer.vocab) # BERT's vocabulary size
vocabulary_size_target = len(tokenizer.vocab) # BERT's vocabulary size
print(vocabulary_size_input, vocabulary_size_target)
embedding_dim = 512
number_layers = 6
number_heads = 8
feed_forward_dim = 2048
my_device = torch.device('cuda:1') if torch.cuda.is_available() else "cpu"
print("DEVICE USED: ", my_device)
model = T4Transformer(vocab_size_input=vocabulary_size_input, vocab_size_target=vocabulary_size_target, d_model=embedding_dim, num_layers=number_layers, 
                        num_heads=number_heads, d_ff=feed_forward_dim, dropout=0.1, pad_token=0, device=my_device, name=None)

loss_fn = CrossEntropyLoss(ignore_index=0)
print("The trainable parameters of the model are: ", count_parameters(model))
model.load_state_dict(torch.load(f'/data/admitosstorage/Phase_2/trained_models/seq/model_t4_combined_e3.pt', map_location=my_device))
print("Okay with loading the model")
# model = DataParallel(model)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
#model = model.to(my_device)
start_token_id = tokenizer.cls_token_id
eos_token_id = tokenizer.sep_token_id

print("Length of Train dataset:", len(train_dataset1))
print("Length of Validation dataset:", len(val_dataset1))
print("Length of Train loader:", len(train_loader1))
print("Length of Validation loader:", len(val_loader1))

xxx = input("Train? [y/n]: ")
if xxx == 'n':
    exit()
else:
    pass

def transform_before_model(batch):
    batch_caption_ids = torch.stack(batch['caption_ids'])#.squeeze(1)
    batch_story_ids = torch.stack(batch['story_ids'])#.squeeze(1)
    #batch_caption_ids = batch_caption_ids.to(my_device)
    #batch_story_ids = batch_story_ids.to(my_device)
    return batch_caption_ids, batch_story_ids
    
num_epochs = 2
accumulation_steps = 1  # Adjust this to control the effective batch size
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=int(num_epochs)*len(train_loader1))

#def train(rank, world_size, my_train_loader, my_val_loader, model, optimizer, scheduler, num_epochs, accumulation_steps):
def train(my_train_loader, my_val_loader, model, optimizer, scheduler, num_epochs, accumulation_steps):
    saver = len(my_train_loader)//10
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar1 = tqdm(my_train_loader, desc=f"Epoch {epoch+1} \n")
        #progress_bar2 = tqdm(train_loader2, desc=f"Epoch {epoch+1} \n")
        #progress_bar3 = tqdm(train_loader3, desc=f"Epoch {epoch+1} \n")
        #for it, (batch1,batch2,batch3) in enumerate(zip(progress_bar1,progress_bar2,progress_bar3)):
        for it, batch1 in enumerate(progress_bar1):
            optimizer.zero_grad()
            # if it < 3370:
            #     continue
            caps1,stories1 = transform_before_model(batch1)
            if caps1.shape[2] + stories1.shape[2] > 300: # and caps1.shape[2] + stories1.shape[2] <= 500 :
                print(f"\n Iteration: {it} with total length shape: {caps1.shape[2] + stories1.shape[2]} ")
                continue
            # else:
            #     continue
            #caps2,stories2 = transform_before_model(batch2)
            #caps3,stories3 = transform_before_model(batch3)
            
            # 1st set of captions
            generated_sequences1 = torch.full((b_s, 1), start_token_id, dtype=torch.long)#.to(my_device)
            iter_loss = 0.0
            for k in range(stories1.shape[2]):
                #print(f"Iteration {k}/{len(my_train_loader)} on form \033[1m 1 \033[0m")
                decoder_input = generated_sequences1#.to(my_device)
                projected1 = model(caps1.squeeze(1), decoder_input)
                probs = F.softmax(projected1[:, -1, :], dim=-1) # put 0 in the 2nd dim for in the 3d 
                next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)#.unsqueeze(-1) 

                #Scheduled sampling
                use_ground_truth = random.random() < (1 - epoch/num_epochs)
                if use_ground_truth:
                    next_token = stories1[:, -1 , k].unsqueeze(-1)

                generated_sequences1 = torch.cat((generated_sequences1, next_token.to('cpu')), dim=1)
                loss1 = loss_fn(projected1[:, -1, :], stories1.squeeze(1)[:, k].to(my_device))
                iter_loss += loss1
                """if k==0:
                    output1 = projected1[:, -1, :] # put 0 in the 2nd dim for in the 3d
                else:
                    torch.cuda.empty_cache()
                    output1 = torch.cat((output1.to('cpu'), projected1[:, -1, :].to('cpu')), 0) # put 0 in the 2nd dim for in the 3d
            if output1.shape[0]!= stories1.view(-1).shape[0]:
                num_rows = output1.size(0)
                random_indices = random.sample(range(num_rows), stories1.view(-1).shape[0])
                output1 = output1[random_indices]                
                loss1 = loss_fn(output1.contiguous().to('cpu'), stories1.contiguous().squeeze(1).view(-1).to('cpu'))
            else:
                loss1 = loss_fn(output1.contiguous().to('cpu'), stories1.contiguous().squeeze(1).view(-1).to('cpu'))"""
    
            # 2ns set of captions
            # generated_sequences2 = torch.full((b_s, 1, 1), start_token_id, dtype=torch.long).to(my_device)
            # for k in range(stories2.shape[2]):
            #     #print(f"Iteration {k}/{len(train_loader2)} on form \033[1m 2 \033[0m ")
            #     decoder_input = generated_sequences2.to(my_device)
            #     projected2 = model(caps2, decoder_input)
            #     probs = F.softmax(projected2[:, 0, -1, :], dim=-1)
            #     next_token = torch.argmax(probs, dim=-1).unsqueeze(-1).unsqueeze(-1) 
            #     generated_sequences2 = torch.cat((generated_sequences2, next_token), dim=2)
            #     if k==0:
            #         output2 = projected2[:, 0, -1, :]
            #     else:
            #         torch.cuda.empty_cache()
            #         output2 = torch.cat((output2.to('cpu'), projected2[:, 0, -1, :].to('cpu')), 0)
            # loss2 = loss_fn(output2.contiguous().to('cpu'), stories2.contiguous().squeeze(1).view(-1).to('cpu'))
    
            # 3rd set of captions 
            # generated_sequences3 = torch.full((b_s, 1, 1), start_token_id, dtype=torch.long).to(my_device)
            # for k in range(stories3.shape[2]):
            #     #print(f"Iteration {k}/{len(train_loader3)} on form \033[1m 3 \033[0m")
            #     decoder_input = generated_sequences3.to(my_device)
            #     projected3 = model(caps3, decoder_input)
            #     probs = F.softmax(projected3[:, 0, -1, :], dim=-1)
            #     next_token = torch.argmax(probs, dim=-1).unsqueeze(-1).unsqueeze(-1) 
            #     generated_sequences3 = torch.cat((generated_sequences3, next_token), dim=2)
            #     if k==0:
            #         output3 = projected3[:, 0, -1, :]
            #     else:
            #         torch.cuda.empty_cache()
            #         output3 = torch.cat((output3.to("cpu"), projected3[:, 0, -1, :].to("cpu")), 0)
            # loss3 = loss_fn(output3.contiguous().to('cpu'), stories3.contiguous().squeeze(1).view(-1).to('cpu'))
    
            """total_loss = loss1 #+ loss2 + loss3
            total_loss.backward()"""
            iter_loss = iter_loss/stories1.size(-1)
            iter_loss.backward()
            if (it+1)%accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
    
            epoch_loss += iter_loss.item()*accumulation_steps
            #avg_loss = iter_loss.item() # / 3  # Average loss from three datasets (if all datasets are used)
            progress_bar1.set_postfix(loss = epoch_loss/(it+1))
            #progress_bar2.set_postfix(loss=avg_loss)
            #progress_bar3.set_postfix(loss=avg_loss)
            print(f'\n ------------------------ Train Iteration {it}/{len(my_train_loader)} completed with running loss --> {iter_loss.item():.10f} ---------------------------------')
            if it%2 == 0:
                torch.cuda.empty_cache()
            if it%saver == 0:
                torch.save(model.state_dict(),f'/data/admitosstorage/Phase_2/trained_models/auto/t4_combined_e{epoch+1}_i{it}.pt')
                
                    
        avg_epoch_loss = epoch_loss / len(my_train_loader)
        print(f'Epoch {epoch+1} completed with train loss: {avg_epoch_loss:.5f}')
        train_loss_per_epoch.append(avg_epoch_loss)
    
        # Validation loop
        print("------------------------ ###### ------------------------ VALIDATION LOOP ------------------------------- ##### -------------------------------")
        model.eval()
        val_epoch_loss = 0.0
        val_progress_bar1 = tqdm(my_val_loader, desc=f"Validation Epoch {epoch+1}")
        #val_progress_bar2 = tqdm(val_loader2, desc=f"Validation Epoch {epoch+1})")
        #val_progress_bar3 = tqdm(val_loader3, desc=f"Validation Epoch {epoch+1}")
        with torch.no_grad():
            #for it, (val_batch1,val_batch2,val_batch3) in enumerate(zip(val_progress_bar1, val_progress_bar2, val_progress_bar3)):
            for it,val_batch1 in enumerate(val_progress_bar1):
                val_caps1, val_stories1 = transform_before_model(val_batch1)
                #val_caps2, val_stories2 = transform_before_model(val_batch2)
                #val_caps3, val_stories3 = transform_before_model(val_batch3)
                val_iter_loss = 0.0 
                generated_sequences1 = torch.full((b_s, 1), start_token_id, dtype=torch.long)#.to(my_device)
                for k in range(val_stories1.shape[2]):
                    #print(f"Iteration {k}/{len(my_val_loader)} on form \033[1m 1 \033[0m")
                    decoder_input = generated_sequences1#.to(my_device)
                    projected1 = model(val_caps1.squeeze(1), decoder_input)
                    probs = F.softmax(projected1[:, -1, :], dim=-1)
                    next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)#.unsqueeze(-1) 
                    generated_sequences1 = torch.cat((generated_sequences1, next_token.to('cpu')), dim=1)
                    val_loss1 = loss_fn(projected1[:, -1, :], val_stories1.squeeze(1)[:, k].to(my_device))
                    val_iter_loss += val_loss1
                    """if k==0:
                        output1 = projected1[:, -1, :]
                    else:
                        torch.cuda.empty_cache()
                        output1 = torch.cat((output1.to('cpu'), projected1[:,  -1, :].to('cpu')), 0)
                #print("Output Final: ", output1.shape)
                #print("Stories: ", val_stories1.shape)  
                if output1.shape[0]!= val_stories1.view(-1).shape[0]:
                    num_rows = output1.size(0)
                    random_indices = random.sample(range(num_rows), val_stories1.view(-1).shape[0])
                    output1 = output1[random_indices]
                    val_loss1 = loss_fn(output1.contiguous().to('cpu'), val_stories1.contiguous().squeeze(1).view(-1).to('cpu'))
                else:
                    val_loss1 = loss_fn(output1.contiguous().to('cpu'), val_stories1.contiguous().squeeze(1).view(-1).to('cpu'))"""
    
                """val_total_loss = val_loss1 # + val_loss2 + val_loss3"""

                val_iter_loss = val_iter_loss/val_stories1.size(-1)
                val_epoch_loss += val_iter_loss.item()
                #val_avg_loss = val_iter_loss.item() #/ 3
                val_progress_bar1.set_postfix(val_loss=val_epoch_loss/(it+1))
                print(f'\n -------------------- Validation Iteration {it}/{len(my_val_loader)} completed with running loss --> {val_iter_loss.item():.10f} -----------------------')
                if it%2 == 0:
                    torch.cuda.empty_cache()
    
        avg_val_epoch_loss = val_epoch_loss / len(my_val_loader)
        print(f'Epoch {epoch+1} completed with validation loss: {avg_val_epoch_loss:.5f}')
        val_loss_per_epoch.append(avg_val_epoch_loss)
        if epoch+1 < 5:
            torch.save(model.state_dict(),f'/data/admitosstorage/Phase_2/trained_models/auto/t4_combined_e{epoch+1}.pt')

    print("Training completed")
    #torch.save(model.state_dict(),'/data/admitosstorage/Phase_2/trained_models/auto/model_t4_combined_final.pt')
    return train_loss_per_epoch, val_loss_per_epoch


all_train_loss, all_val_loss = train(train_loader1, val_loader1, model, optimizer, scheduler, num_epochs, accumulation_steps)
epochs = range(1, len(all_train_loss) + 1)
plt.plot(epochs, all_train_loss, color='blue', label='Train Loss')
plt.plot(epochs, all_val_loss, color='orange', label='Validation Loss')
plt.title('Train/Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend()
#plt.savefig(f'./plots/phase2/seq/train_val_loss_t4_combined_v1.png')
