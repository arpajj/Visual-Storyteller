import torch, pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

t5_path_train_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/v2_train_caps_t5y.pkl'
t5_path_train_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/v2_train_stories_t5y.pkl'

with open(t5_path_train_a, 'rb') as f:
    train_caps_encodings = pickle.load(f)[:-1]

with open(t5_path_train_b, 'rb') as f:
    train_story_labels = pickle.load(f)[:-1]

t5_path_val_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/v2_val_caps_t5y.pkl'
t5_path_val_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/v2_val_stories_t5y.pkl'

with open(t5_path_val_a, 'rb') as f:
    val_caps_encodings = pickle.load(f)[:-1]

with open(t5_path_val_b, 'rb') as f:
    val_story_labels = pickle.load(f)[:-1]

t5_path_test_a = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/v2_test_caps_t5y.pkl'
t5_path_test_b = f'/data/admitosstorage/Phase_2/data_phase2/T5/Yingjin/v2_test_stories_t5y.pkl'

with open(t5_path_test_a, 'rb') as f:
    test_caps_encodings = pickle.load(f)[:-1]

with open(t5_path_test_b, 'rb') as f:
    test_story_labels = pickle.load(f)[:-1]


class StoryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val.clone().detach() for key, val in self.encodings[idx].items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)
    
def my_collate_fn(batch):
    input_ids = [item['input_ids'].transpose(0,1) for item in batch]
    attention_mask = [item['attention_mask'].transpose(0,1) for item in batch]
    labels = [item['labels'].transpose(0,1) for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    final_input_ids = [item.transpose(0,1) for item in input_ids_padded]
    final_attention_masks = [item.transpose(0,1) for item in attention_mask_padded]
    final_labels = [item.transpose(0,1) for item in labels_padded]

    return {'input_ids': final_input_ids, 'attention_mask': final_attention_masks, 'labels': final_labels}


train_dataset = StoryDataset(train_caps_encodings, train_story_labels)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=my_collate_fn)

val_dataset = StoryDataset(val_caps_encodings, val_story_labels)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True, collate_fn=my_collate_fn)

test_dataset = StoryDataset(test_caps_encodings, test_story_labels)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, collate_fn=my_collate_fn)


model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

print("The |V| is: ", len(tokenizer.get_vocab()))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("The parametrs of the model are:", count_parameters(model))
my_device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"
print("DEVICE USED: ", my_device)
optimizer = AdamW(model.parameters(), lr=5e-5, no_deprecation_warning=True)
model = model.to(my_device)

num_epochs = 50
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=int(num_epochs)*len(train_loader))

xxx = input("Train? [y/n]: ")
if xxx == 'n':
    exit()
else:
    pass

train_loss_per_epoch = []
val_loss_per_epoch = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for it, batch in enumerate(progress_bar):
        model.zero_grad()
        batch_input_ids = torch.stack(batch['input_ids']).squeeze(1)
        batch_att_masks = torch.stack(batch['attention_mask']).squeeze(1)
        batch_labels = torch.stack(batch['labels']).squeeze(1)
        final_labels = batch_labels.clone()
        final_labels[batch_labels == tokenizer.pad_token_id] = -100

        batch_input_ids = batch_input_ids.to(my_device)
        batch_att_masks = batch_att_masks.to(my_device)
        final_labels = final_labels.to(my_device)    

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_att_masks, labels=final_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        print(f'Train Iteration {it} completed with running loss: {loss.item()}')
        if it%10==0:
            torch.cuda.empty_cache()

    print(f'Epoch {epoch} completed with train loss: {train_loss/len(train_loader)}')
    train_loss_per_epoch.append(train_loss/len(train_loader))

    model.eval()
    val_loss = 0.0
    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
    with torch.no_grad():
        for it, batch in enumerate(progress_bar):
            batch_input_ids = torch.stack(batch['input_ids']).squeeze(1)
            batch_att_masks = torch.stack(batch['attention_mask']).squeeze(1)
            batch_labels = torch.stack(batch['labels']).squeeze(1)
            final_labels = batch_labels.clone()
            final_labels[batch_labels == tokenizer.pad_token_id] = -100

            batch_input_ids = batch_input_ids.to(my_device)
            batch_att_masks = batch_att_masks.to(my_device)
            final_labels = final_labels.to(my_device)
            
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_att_masks, labels=final_labels)
            val_loss += outputs.loss.item()
            progress_bar.set_postfix(loss=outputs.loss.item())
            print(f'Val Iteration {it} completed with running loss: {outputs.loss.item()}')
            if it%10==0:
                torch.cuda.empty_cache()


    print(f"Validation Loss: {val_loss/len(val_loader)}")
    val_loss_per_epoch.append(val_loss/len(val_loader))
    if (epoch+1)%10==0:
        torch.save(model.state_dict(),f'/data/admitosstorage/Phase_2/trained_models/T5/v2_e100_model_t5_yingjin_{(epoch+1)//10}.pt')


torch.save(model.state_dict(),'/data/admitosstorage/Phase_2/trained_models/T5/model_t5_yingjin_5x1_v2.pt')

epochs = range(1, len(train_loss_per_epoch) + 1)
plt.plot(epochs, train_loss_per_epoch, color='blue', label='Train Loss')
plt.plot(epochs, val_loss_per_epoch, color='orange', label='Validation Loss')
plt.title('Train/Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend()
plt.savefig(f'./plots/phase2/t5/train_val_loss_t5_v2.png')



