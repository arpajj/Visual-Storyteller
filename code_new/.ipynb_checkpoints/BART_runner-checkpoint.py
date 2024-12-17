from transformers import BartTokenizer, BartForConditionalGeneration
import torch, random, pickle
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

path_train_a = f'/data/admitosstorage/Phase_2/data_phase2/BART/train_caps_1.pkl'
path_train_b = f'/data/admitosstorage/Phase_2/data_phase2/BART/train_caps_2.pkl'
path_train_c = f'/data/admitosstorage/Phase_2/data_phase2/BART/train_caps_3.pkl'

with open(path_train_a, 'rb') as f:
    tokenized_train_captions1 = pickle.load(f)

with open(path_train_b, 'rb') as f:
    tokenized_train_captions2 = pickle.load(f)

with open(path_train_c, 'rb') as f:
    tokenized_train_captions3 = pickle.load(f)

path_val_a = f'/data/admitosstorage/Phase_2/data_phase2/BART/val_caps_1.pkl'
path_val_b = f'/data/admitosstorage/Phase_2/data_phase2/BART/val_caps_2.pkl'
path_val_c = f'/data/admitosstorage/Phase_2/data_phase2/BART/val_caps_3.pkl'

with open(path_val_a, 'rb') as f:
    tokenized_val_captions1 = pickle.load(f)

with open(path_val_b, 'rb') as f:
    tokenized_val_captions2 = pickle.load(f)

with open(path_val_c, 'rb') as f:
    tokenized_val_captions3 = pickle.load(f)
    
path_test_a = f'/data/admitosstorage/Phase_2/data_phase2/BART/test_caps_1.pkl'
path_test_b = f'/data/admitosstorage/Phase_2/data_phase2/BART/test_caps_2.pkl'
path_test_c = f'/data/admitosstorage/Phase_2/data_phase2/BART/test_caps_3.pkl'

with open(path_test_a, 'rb') as f:
    tokenized_test_captions1 = pickle.load(f)

with open(path_test_b, 'rb') as f:
    tokenized_test_captions2 = pickle.load(f)

with open(path_test_c, 'rb') as f:
    tokenized_test_captions3 = pickle.load(f)

path_train_stories = f'/data/admitosstorage/Phase_2/data_phase2/BART/train_stories.pkl'
path_val_stories = f'/data/admitosstorage/Phase_2/data_phase2/BART/val_stories.pkl'
path_test_stories = f'/data/admitosstorage/Phase_2/data_phase2/BART/test_stories.pkl'

with open(path_train_stories, 'rb') as f:
    tokenized_train_stories = pickle.load(f)

with open(path_val_stories, 'rb') as f:
    tokenized_val_stories = pickle.load(f)

with open(path_test_stories, 'rb') as f:
    tokenized_test_stories = pickle.load(f)


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

b_s = 8
train_dataset = StoryDataset(tokenized_train_captions1, tokenized_train_stories)
train_loader = DataLoader(train_dataset, batch_size=b_s, shuffle=True, collate_fn=my_collate_fn)

val_dataset = StoryDataset(tokenized_val_captions1, tokenized_val_stories)
val_loader = DataLoader(val_dataset, batch_size=b_s, shuffle=True, collate_fn=my_collate_fn)

test_dataset = StoryDataset(tokenized_test_captions1, tokenized_test_stories)
test_loader = DataLoader(test_dataset, batch_size=b_s, shuffle=True, collate_fn=my_collate_fn)


for batch in train_loader:
    print("Captions shape inner:", batch['input_ids'][0].shape, batch['input_ids'][1].shape, batch['input_ids'][2].shape, batch['input_ids'][3].shape )
    print("Attention shape: ", batch['attention_mask'][0].shape, batch['attention_mask'][1].shape)
    print("Stories shape:", batch['labels'][0].shape,  batch['labels'][1].shape)
    break

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize the model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

print("The |V| is: ", len(tokenizer.get_vocab()))

print("The parametrs of the model are:", count_parameters(model))
my_device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"
print("DEVICE USED: ", my_device)
model = model.to(my_device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 12
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_epochs*len(train_loader))

xxx = input("Train? [y/n]: ")
if xxx == 'n':
    exit()
else:
    pass

# Training function
def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs):
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")

        for it,batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_input_ids = torch.stack(batch['input_ids']).squeeze(1)
            batch_input_ids = batch_input_ids.to(my_device)
            batch_att_masks = torch.stack(batch['attention_mask']).squeeze(1) 
            batch_att_masks = batch_att_masks.to(my_device)
            batch_labels = torch.stack(batch['labels']).squeeze(1)
            batch_labels = batch_labels.to(my_device)
            if it==0:
                print(batch_input_ids.shape)
                print(batch_att_masks.shape)
                print(batch_labels.shape)
            
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_att_masks, labels=batch_labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            print(f'Train Iteration {it} completed with running loss: {loss.item()}')
            torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_per_epoch.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.5f}')

        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for batch in val_progress_bar:
                batch_input_ids = torch.stack(batch['input_ids']).squeeze(1).to(my_device)
                batch_att_masks = torch.stack(batch['attention_mask']).squeeze(1).to(my_device)
                batch_labels = torch.stack(batch['labels']).squeeze(1).to(my_device)


                outputs = model(input_ids=batch_input_ids, attention_mask=batch_att_masks, labels=batch_labels)
                val_loss += outputs.loss.item()
                print(f'Val Iteration {it} completed with running loss: {outputs.loss.item()}')
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_per_epoch.append(avg_val_loss)
        print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.5f}')
        if (epoch+1)%2==0:
            torch.save(model.state_dict(),f'/data/admitosstorage/Phase_2/trained_models/BART/trained_bart_e{epoch}.pt')

    print("Training completed")
    return train_loss_per_epoch, val_loss_per_epoch

# Train the model
train_loss_all, val_loss_all = train(model, train_loader, val_loader, optimizer, scheduler, num_epochs)

epochs = range(1, len(train_loss_all) + 1)
plt.plot(epochs, train_loss_all, color='blue', label='Train Loss')
plt.plot(epochs, val_loss_all, color='orange', label='Validation Loss')
plt.title('Train/Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend()
plt.savefig(f'./plots/phase2/bart/train_val_loss_bart.png')
