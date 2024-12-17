import torch, pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

print("Please choose the type of the strategy that you want between 'admitos' and 'yingjin'")
while True:
    name = input()
    print("You have chosen:", name.upper())
    if name.lower() == 'admitos' or name.lower() == 'yingjin':
        break
    else:
        print("Please choose only one of the 2 strategies 'admitos' or 'yingjin'")

class CustomDatasetT5(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_encodings = self.input_data[idx]
        target_encodings = self.target_data[idx]

        input_ids = input_encodings['input_ids'].clone().detach()
        attention_mask = input_encodings['attention_mask'].clone().detach()

        target_ids = target_encodings['input_ids'].clone().detach()
        target_attention_mask = target_encodings['attention_mask'].clone().detach()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target_ids': target_ids, 'target_attention_mask': target_attention_mask}
    
def my_collate_fn(batch):
    input_ids = [item['input_ids'].transpose(0,1) for item in batch]
    attention_mask = [item['attention_mask'].transpose(0,1) for item in batch]
    target_ids = [item['target_ids'].transpose(0,1) for item in batch]
    target_attention_mask = [item['target_attention_mask'].transpose(0,1) for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    target_attention_mask_padded = pad_sequence(target_attention_mask, batch_first=True, padding_value=0)

    final_input_ids = [item.transpose(0,1) for item in input_ids_padded]
    final_attention_masks = [item.transpose(0,1) for item in attention_mask_padded]
    final_target_ids = [item.transpose(0,1) for item in target_ids_padded]
    final_target_attention_masks = [item.transpose(0,1) for item in target_attention_mask_padded]

    return {'input_ids': final_input_ids, 'attention_mask': final_attention_masks, 'target_ids': final_target_ids, 'target_attention_mask': final_target_attention_masks}

if name.lower() == 'admitos':
    t5_path_train_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Admitos/train_caps_t5a.pkl'
    t5_path_train_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Admitos/train_stories_t5a.pkl'
    
    with open(t5_path_train_a, 'rb') as f:
        toked_train_caps = pickle.load(f)
    
    with open(t5_path_train_b, 'rb') as f:
        toked_train_stories = pickle.load(f)
    
    t5_path_val_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Admitos/val_caps_t5a.pkl'
    t5_path_val_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Admitos/val_stories_t5a.pkl'
    
    with open(t5_path_val_a, 'rb') as f:
        toked_val_caps = pickle.load(f)
    
    with open(t5_path_val_b, 'rb') as f:
        toked_val_stories = pickle.load(f)
    
    t5_path_test_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Admitos/test_caps_t5a.pkl'
    t5_path_test_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Admitos/test_stories_t5a.pkl'
    
    with open(t5_path_test_a, 'rb') as f:
        toked_test_caps = pickle.load(f)
    
    with open(t5_path_test_b, 'rb') as f:
        toked_test_stories = pickle.load(f)


if name.lower() == 'yingjin':
    t5_path_train_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/train_caps_t5y.pkl'
    t5_path_train_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/train_stories_t5y.pkl'
    
    with open(t5_path_train_a, 'rb') as f:
        toked_train_caps = pickle.load(f)
    
    with open(t5_path_train_b, 'rb') as f:
        toked_train_stories = pickle.load(f)
    
    t5_path_val_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/val_caps_t5y.pkl'
    t5_path_val_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/val_stories_t5y.pkl'
    
    with open(t5_path_val_a, 'rb') as f:
        toked_val_caps = pickle.load(f)
    
    with open(t5_path_val_b, 'rb') as f:
        toked_val_stories = pickle.load(f)
    
    t5_path_test_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/test_caps_t5y.pkl'
    t5_path_test_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/test_stories_t5y.pkl'
    
    with open(t5_path_test_a, 'rb') as f:
        toked_test_caps = pickle.load(f)
    
    with open(t5_path_test_b, 'rb') as f:
        toked_test_stories = pickle.load(f)


b_s = 5
final_train_dataset = CustomDatasetT5(toked_train_caps, toked_train_stories)
final_train_dataloader = DataLoader(final_train_dataset, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

final_val_dataset = CustomDatasetT5(toked_val_caps, toked_val_stories)
final_val_dataloader = DataLoader(final_val_dataset, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

final_test_dataset = CustomDatasetT5(toked_test_caps, toked_test_stories)
final_test_dataloader = DataLoader(final_test_dataset, batch_size=b_s, shuffle=False, collate_fn=my_collate_fn)

for batch in final_train_dataloader:
    print((batch['input_ids'][0].shape))
    print(len(batch['attention_mask']))
    print((batch['target_ids'][0].shape))
    print(len(batch['target_attention_mask']))
    break

model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("The parametrs of the model are:", count_parameters(model))
my_device = torch.device('cuda:1') if torch.cuda.is_available() else "cpu"
print("DEVICE USED: ", my_device)
optimizer = AdamW(model.parameters(), lr=5e-5, no_deprecation_warning=True)
model = model.to(my_device)


train_loss_per_10_steps = []
val_loss_per_10_steps = []

num_epochs = 10
for epoch in range(num_epochs):
    # ------------------------------------------------------------- Training loop -----------------------------------------------
    model.train()
    print('Running epoch: {}'.format(epoch))
    running_train_loss = 0.0
    progress_bar = tqdm(final_train_dataloader, desc=f"Epoch {epoch+1}")
    for idx, batch in enumerate(final_train_dataloader):
        # clear out the gradients of all Variables 
        batch_input_ids = torch.stack(batch['input_ids'])
        batch_att_masks = torch.stack(batch['attention_mask'])
        batch_target_ids = torch.stack(batch['target_ids'])
        batch_tar_att_masks = torch.stack(batch['target_attention_mask'])

        batch_size, num_captions, seq_len_cap = batch_input_ids.size()
        batch_size, num_story_sents, seq_len_story = batch_target_ids.size()

        batch_input_ids = batch_input_ids.view(batch_size, seq_len_cap*num_captions)
        batch_att_masks = batch_att_masks.view(batch_size, seq_len_cap*num_captions)
        batch_target_ids = batch_target_ids.view(batch_size, seq_len_story*num_story_sents)
        batch_tar_att_masks = batch_tar_att_masks.view(batch_size, seq_len_story*num_story_sents)

        labels = batch_target_ids.contiguous()
        optimizer.zero_grad()
        
        # padding_token_id = tokenizer.pad_token_id
        # labels[batch_target_ids == padding_token_id] = -100  # We only compute loss on non-pad
        batch_input_ids = batch_input_ids.to(my_device)
        batch_att_masks = batch_att_masks.to(my_device)
        batch_target_ids = batch_target_ids.to(my_device)
        labels = labels.to(my_device)

        outputs = model(input_ids = batch_input_ids, attention_mask = batch_att_masks, decoder_input_ids = batch_target_ids, labels = labels)
        train_loss = outputs.loss
        
        # calculating the gradients
        train_loss.backward()
        optimizer.step()

        print('Iteration: {}, Running Train loss: {}'.format(idx, train_loss.item()))
        running_train_loss += train_loss.item()
        if idx%10 == 0:      
            train_loss_per_10_steps.append(train_loss.item())
            torch.cuda.empty_cache()

        progress_bar.set_postfix(loss=train_loss.item())
        progress_bar.update()

    progress_bar.close()
    running_train_loss = running_train_loss/int(len(final_train_dataloader))
    print('Epoch: {}, Train loss: {}'.format(epoch, running_train_loss))
    
    # -------------------------------------------- Validation loop ---------------------------------------------
    model.eval()
    running_val_loss = 0.0
    progress_bar = tqdm(total=len(final_val_dataloader), desc=f"Epoch {epoch+1}")
    with torch.no_grad():
        for idx, batch in enumerate(final_val_dataloader):
            batch_input_ids = torch.stack(batch['input_ids'])
            batch_att_masks = torch.stack(batch['attention_mask'])
            batch_target_ids = torch.stack(batch['target_ids'])
            batch_tar_att_masks = torch.stack(batch['target_attention_mask'])

            batch_size, num_captions, seq_len_cap = batch_input_ids.size()
            batch_size, num_story_sents, seq_len_story = batch_target_ids.size()

            batch_input_ids = batch_input_ids.view(batch_size, seq_len_cap*num_captions)
            batch_att_masks = batch_att_masks.view(batch_size, seq_len_cap*num_captions)
            batch_target_ids = batch_target_ids.view(batch_size, seq_len_story*num_story_sents)
            batch_tar_att_masks = batch_tar_att_masks.view(batch_size, seq_len_story*num_story_sents)

            labels = batch_target_ids.contiguous()
            
            batch_input_ids = batch_input_ids.to(my_device)
            batch_att_masks = batch_att_masks.to(my_device)
            batch_target_ids = batch_target_ids.to(my_device)
            labels = labels.to(my_device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_att_masks, decoder_input_ids=batch_target_ids, labels=labels)
            val_loss = outputs.loss

            print('Iteration: {}, Running Val loss: {}'.format(idx, val_loss.item()))
            running_val_loss += val_loss.item()
            if idx%10 ==0:      
                val_loss_per_10_steps.append(val_loss.item())
                torch.cuda.empty_cache()
            progress_bar.set_postfix(loss=val_loss.item())
            progress_bar.update()
            
    progress_bar.close()
    running_val_loss = running_val_loss / int(len(final_val_dataloader))
    print('Epoch: {}, Validation loss: {}'.format(epoch, running_val_loss))

torch.save(model.state_dict(),'/data/admitosstorage/Phase_2/trained_models/T5/model_t5_admitos_5x3.pt')