""" ######################## HERE ITS T4_RUNNER ################################"""
import pickle, torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from t4 import T4Transformer, T4tokenizer
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from torch.nn import DataParallel

print("Please choose the type of the strategy that you want between 'admitos' and 'yingjin'")
while True:
    name = input()
    print("You have chosen:", name.upper())
    if name.lower() == 'admitos' or name.lower() == 'yingjin':
        break
    else:
        print("Please choose only one of the 2 strategies 'admitos' or 'yingjin'")

print("Should we use BERT as Tokenizer? (answer with 'yes' or 'no')")
str_bert = input()
print("You have chosen:", str_bert.upper())
if str_bert.lower() == 'yes':
    use_bert = True
else:
    use_bert = False

### FUNCTIONS
class CustomDataset(Dataset):
    def __init__(self, entries, references):
        self.entries = entries
        self.references = references

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        reference = self.references[idx]
        return entry, reference


def custom_collate_fn(batch):
    entries, references = zip(*batch)
    
    # Determine the max length in the batch for entries and references
    max_len_entry = max(entry.size(-1) for entry in entries)
    max_len_ref = max(ref.size(-1) for ref in references)
    
    # Pad entries
    padded_entries = [F.pad(entry, (0, max_len_entry - entry.size(-1))) for entry in entries]
    padded_entries = torch.stack(padded_entries, dim=0)
    
    # Pad references
    padded_references = [F.pad(ref, (0, max_len_ref - ref.size(-1))) for ref in references]
    padded_references = torch.stack(padded_references, dim=0)
    
    return padded_entries, padded_references

def compute_loss(output, target):
    loss_fn = CrossEntropyLoss(ignore_index=0)
    loss = loss_fn(output, target)
    return loss

def flat_over_img_caps(source):
    if len(source.shape)<3:
        return source
    if len(source.shape)==3:
        if (source.shape[0]==5 and source.shape[1]==3) or (source.shape[0]==5 and source.shape[1]==2):
            source = source.reshape(-1, source.size(-1)) 
            return source
        else:
            return source
    if len(source.shape)==4:
        source = source.reshape(source.size(0), -1, source.size(-1))
        return source
    assert len(source.shape)>4, f"AssertionError: The length of the source is more than 4!"
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_losses(train_losses, validation_losses, epoch=None):
    plt.clf()
    if epoch != None:
        iterations = range(1, len(train_losses) + 1)
        plt.plot(iterations, train_losses, color='blue', label='Train Loss')
        plt.plot(iterations, validation_losses, color='orange', label='Validation Loss')
        plt.title(f'Train/Validation Loss Over Iterations for epoch: {epoch}')
        plt.xlabel('Iteration')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(f'./plots/phase2/auto/train_val_loss_plot_e{epoch+1}.png')
    else:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, color='blue', label='Train Loss')
        plt.plot(epochs, validation_losses, color='orange', label='Validation Loss')
        plt.title('Train/Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(f'./plots/phase2/seq/train_val_loss_plot.png')

## Admitos Way with BERT Tokenizer
if name.lower() == 'admitos' and use_bert:
    phase2_path_train_src = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/final_input_train_data.pkl'
    phase2_path_train_tgt = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/final_ref_train_data.pkl'
    
    with open(phase2_path_train_src, 'rb') as f:
        train_source_caps = pickle.load(f)
    
    with open(phase2_path_train_tgt, 'rb') as f:
        train_target_stories = pickle.load(f)
    
    phase2_path_val_src = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/final_input_val_data.pkl'
    phase2_path_val_tgt = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/final_ref_val_data.pkl'
    
    with open(phase2_path_val_src, 'rb') as f:
        val_source_caps = pickle.load(f)
    
    with open(phase2_path_val_tgt, 'rb') as f:
        val_target_stories = pickle.load(f)
    
    phase2_path_test_src = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/final_input_test_data.pkl'
    phase2_path_test_tgt = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/final_ref_test_data.pkl'
    
    with open(phase2_path_test_src, 'rb') as f:
        test_source_caps = pickle.load(f)
    
    with open(phase2_path_test_tgt, 'rb') as f:
        test_target_stories = pickle.load(f)

## Admitos Way with Our Tokenizer
if name.lower() == 'admitos' and not use_bert:
    phase2_path_train_src = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/train_caps_t4tok.pkl'
    phase2_path_train_tgt = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/train_stories_t4tok.pkl'
    
    with open(phase2_path_train_src, 'rb') as f:
        train_source_caps = pickle.load(f)
    
    with open(phase2_path_train_tgt, 'rb') as f:
        train_target_stories = pickle.load(f)
    
    phase2_path_val_src = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/val_caps_t4tok.pkl'
    phase2_path_val_tgt = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/val_stories_t4tok.pkl'
    
    with open(phase2_path_val_src, 'rb') as f:
        val_source_caps = pickle.load(f)
    
    with open(phase2_path_val_tgt, 'rb') as f:
        val_target_stories = pickle.load(f)
        
    phase2_path_test_src = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/test_caps_t4tok.pkl'
    phase2_path_test_tgt = '/data/admitosstorage/Phase_2/data_phase2/Admitos_way/test_stories_t4tok.pkl'
    
    with open(phase2_path_test_src, 'rb') as f:
        test_source_caps = pickle.load(f)
    
    with open(phase2_path_test_tgt, 'rb') as f:
        test_target_stories = pickle.load(f)

## Yingjin Way with BERT Tokenizer
if name.lower() == 'yingjin' and use_bert:
    phase2_path_yingjin_train_a = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/final_train_input_caps.pkl'
    phase2_path_yingjin_train_b = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/final_train_ref_stories.pkl'
    
    with open(phase2_path_yingjin_train_a, 'rb') as f:
        train_source_caps = pickle.load(f)
    
    with open(phase2_path_yingjin_train_b, 'rb') as f:
        train_target_stories = pickle.load(f)
    
    phase2_path_yingjin_val_a = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/final_val_input_caps.pkl'
    phase2_path_yingjin_val_b = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/final_val_ref_stories.pkl'
    
    with open(phase2_path_yingjin_val_a, 'rb') as f:
        val_source_caps = pickle.load(f)
    
    with open(phase2_path_yingjin_val_b, 'rb') as f:
        val_target_stories = pickle.load(f)
    
    phase2_path_yingjin_test_a = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/final_test_input_caps.pkl'
    phase2_path_yingjin_test_b = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/final_test_ref_stories.pkl'
    
    with open(phase2_path_yingjin_test_a, 'rb') as f:
        test_source_caps = pickle.load(f)
    
    with open(phase2_path_yingjin_test_b, 'rb') as f:
        test_target_stories = pickle.load(f)

## Yingjin Way with Our Tokenizer
if name.lower() == 'yingjin' and not use_bert:
    phase2_path_yingjin_train_a = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/train_caps_t4tok.pkl'
    phase2_path_yingjin_train_b = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/train_stories_t4tok.pkl'
    
    with open(phase2_path_yingjin_train_a, 'rb') as f:
        train_source_caps = pickle.load(f)
    
    with open(phase2_path_yingjin_train_b, 'rb') as f:
        train_target_stories = pickle.load(f)
    
    phase2_path_yingjin_val_a = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/val_caps_t4tok.pkl'
    phase2_path_yingjin_val_b = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/val_stories_t4tok.pkl'
    
    with open(phase2_path_yingjin_val_a, 'rb') as f:
        val_source_caps = pickle.load(f)
    
    with open(phase2_path_yingjin_val_b, 'rb') as f:
        val_target_stories = pickle.load(f)
        
    phase2_path_yingjin_test_a = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/test_caps_t4tok.pkl'
    phase2_path_yingjin_test_b = '/data/admitosstorage/Phase_2/data_phase2/Yingjin_way/test_stories_t4tok.pkl'
    
    with open(phase2_path_yingjin_test_a, 'rb') as f:
        test_source_caps = pickle.load(f)
    
    with open(phase2_path_yingjin_test_b, 'rb') as f:
        test_target_stories = pickle.load(f)

## Load the neccesary vocabs
if use_bert == False:
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

## Create the Datasets and the DataLoaders
b_s = 20 # set the batch size

final_train_dataset = CustomDataset(train_source_caps, train_target_stories)
final_val_dataset = CustomDataset(val_source_caps, val_target_stories)
final_test_dataset = CustomDataset(test_source_caps, test_target_stories)

final_train_dataloader = DataLoader(final_train_dataset, batch_size=b_s, shuffle=True, collate_fn=custom_collate_fn)
final_val_dataloader = DataLoader(final_val_dataset, batch_size=b_s, shuffle=True, collate_fn=custom_collate_fn)
final_test_dataloader = DataLoader(final_test_dataset, batch_size=b_s, shuffle=True, collate_fn=custom_collate_fn)

## Check the sizes of the inputs 
print("Train lenth:", len(final_train_dataloader))
print("Validation lenth:", len(final_val_dataloader))
print("Test lenth:", len(final_test_dataloader))
print()

all_dim3 = []
for x1,x2 in final_train_dataloader:
    all_dim3.append(x2.shape[2])

for x1,x2 in final_train_dataloader:
    print(x1.shape)
    print(x2.shape)
    break
    
print()
for x1,x2 in final_val_dataloader:
    print(x1.shape)
    print(x2.shape)
    break

print()
for x1,x2 in final_test_dataloader:
    print(x1.shape)
    print(x2.shape)
    break


if use_bert:
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    vocabulary_size_input = len(tokenizer.vocab) # BERT's vocabulary size
    vocabulary_size_target = len(tokenizer.vocab) # BERT's vocabulary size
    
else:
    tokenizer = T4tokenizer(total_vocab)
    vocabulary_size_input = len(tokenizer.get_vocab()) # number of unique tokens in all of Inputs
    vocabulary_size_target = len(tokenizer.get_vocab()) # number of unique tokens in all of Targets

print(f"The length of Input and Ouput Vocabularies are {vocabulary_size_input} and {vocabulary_size_target} respectively")

embedding_dim = 512
number_layers = 4
number_heads = 8
feed_forward_dim = 1024
my_device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"
print("DEVICE USED: ", my_device)
model = T4Transformer(vocab_size_input=vocabulary_size_input, vocab_size_target=vocabulary_size_target, d_model=embedding_dim, num_layers=number_layers, 
                        num_heads=number_heads, d_ff=feed_forward_dim, dropout=0.2, pad_token=0, device=my_device, name=name)

optimizer = optim.Adam(model.parameters(), lr=0.001)
print("The trainable parameters of the model are: ", count_parameters(model))

type_of_training = input("Please choose the type of training between 'seq' and 'auto': ")
print(f"You have entered {type_of_training.upper()}")
number_epochs = input("Please enter the number of epochs: ")
print(f"You have entered {number_epochs} epochs")
#model = DataParallel(model)
model = model.to(my_device)

# train sequence-to-sequence 
def train_seq2seq(num_epochs, scheduler):
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    for epoch in range(num_epochs):
        print("------------------------------------------ Training loop ------------------------------------------------")
        epoch_train_loss = 0.0
        model.train()  # Set model to training mode
        #accumulation_steps = 2 
        for it, (input_data, target_data) in tqdm(enumerate(final_train_dataloader), total=len(final_train_dataloader),  desc="Processing Train items", ncols=100):
            #src = input_data[:, :, 0:2, :] # For 4D input data
            src = input_data
            tgt = target_data
            if src.shape[0] == 1:
               src = src.squeeze(0) # Remove batch dimension
               tgt = tgt.squeeze(0) # Remove batch dimension
    
            src = flat_over_img_caps(src)
            model.zero_grad()
            output = model(src.to(my_device), tgt.to(my_device))
            print(output.view(-1,vocabulary_size_target).shape)
            print(tgt.squeeze(0).view(-1).shape)
            train_loss = compute_loss(output.contiguous().view(-1,vocabulary_size_target), tgt.squeeze(0).contiguous().view(-1).to(my_device))
            train_loss.backward()
            #if (it+1)%accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch+1}, Iteration [{it+1}/{len(final_train_dataloader)}] --> Train Loss: {train_loss.item()}")
            epoch_train_loss += train_loss.item()
            if it%5==0:
                torch.cuda.empty_cache()
    
        avg_train_loss = epoch_train_loss/len(final_train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}")
        train_loss_per_epoch.append(avg_train_loss)
    
        print("------------------------------------------ Validation loop ------------------------------------------------")
        epoch_val_loss = 0.0
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():  # Disable gradient calculation for validation
            for it, (input_data, target_data) in tqdm(enumerate(final_val_dataloader), total=len(final_val_dataloader), desc="Processing Validation items", ncols=100):
                #src = input_data[:, :, 0:2, :] # For 4D input data
                src = input_data
                tgt = target_data
                if src.shape[0] == 1:
                    src = src.squeeze(0)  # Remove batch dimension
                    tgt = tgt.squeeze(0)  # Remove batch dimension
    
                src = flat_over_img_caps(src)
                output = model(src.to(my_device), tgt.to(my_device))
    
                val_loss = compute_loss(output.contiguous().view(-1, vocabulary_size_target), tgt.squeeze(0).contiguous().view(-1).to(my_device))
                print(f"Epoch: {epoch+1}, Iteration [{it+1}/{len(final_val_dataloader)}] --> Validation Loss: {val_loss.item()}")
                epoch_val_loss += val_loss.item()
                if it%5==0:
                    torch.cuda.empty_cache()
        
        avg_val_loss = epoch_val_loss/len(final_val_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Val Loss: {avg_val_loss:.4f}")
        val_loss_per_epoch.append(avg_val_loss)
        
        
    print()
    print("Training finished!")
    
    plot_losses(train_loss_per_epoch,val_loss_per_epoch)
    torch.save(model.state_dict(),'/data/admitosstorage/Phase_2/trained_models/seq/model_bert_yingjin_5x1.pt')

### ------------------------------------------------------------------------------------------------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------------------------------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------------------------------------------------------------------------------------------------ ###

# train autoregressively 
def train_auto(num_epochs, scheduler):
    start_token_id = tokenizer.cls_token_id
    eos_token_id = tokenizer.sep_token_id
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    for epoch in range(num_epochs):
        print("------------------------------------------ Training loop ------------------------------------------------")
        epoch_train_loss = 0.0
        model.train()  # Set model to training mode
        
        train_loss_per_iter = []
        for it, (input_data, target_data) in tqdm(enumerate(final_train_dataloader), total=len(final_train_dataloader),  desc="Processing Train items", ncols=100):
            #src = input_data[:, :, 0:2, :] # For 4D input data
            src = input_data
            tgt = target_data
            if src.shape[0] == 1:
               src = src.squeeze(0)  # Remove batch dimension
               tgt = tgt.squeeze(0)  # Remove batch dimension
    
            src = flat_over_img_caps(src)
            generated_sequences = torch.full((tgt.size(0), 1, 1), start_token_id, dtype=torch.long)
            running_train_loss = 0.0
            for k in range(tgt.size(2)):
                model.zero_grad()
                decoder_input =  tgt[:, :, 0:k+1]
                output = model(src.to(my_device), decoder_input.to(my_device))
                train_loss = compute_loss(output.contiguous().view(-1,vocabulary_size_target), decoder_input.squeeze(0).contiguous().view(-1).to(my_device))
                train_loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print(f"Epoch: {epoch+1}, Iteration: {it+1}/{len(final_train_dataloader)}, Generation step: {k+1}/{tgt.size(2)}, Loss: {train_loss.item():.5f}")
                running_train_loss += train_loss.item()
                if k%10 == 0:
                    torch.cuda.empty_cache()
                    
            avg_running_train_loss = running_train_loss/tgt.size(2)
            print(f"Epoch: {epoch+1}, Iteration {it+1}/{len(final_train_dataloader)}, Loss: {avg_running_train_loss:.5f}")
            train_loss_per_iter.append(avg_running_train_loss)
            epoch_train_loss += avg_running_train_loss

            if it%130==0 and it>=130:
                saver = int(it//130)
                torch.save(model.state_dict(), f'/data/admitosstorage/Phase_2/trained_models/auto/model_bert_yingjin_e{epoch+1}_v{saver}.pt')
            
        avg_epoch_train_loss = epoch_train_loss/len(final_train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_train_loss:.5f}")
        #train_loss_per_epoch.append(avg_train_loss)
        train_loss_per_epoch.append(train_loss_per_iter)
        
        
        print("------------------------------------------ Validation loop ------------------------------------------------")
        epoch_val_loss = 0.0
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():  # Disable gradient calculation for validation
            val_loss_per_iter = []
            for it, (input_data, target_data) in tqdm(enumerate(final_val_dataloader),total=len(final_val_dataloader), desc="Processing Validation items", ncols=100):
                #src = input_data[:, :, 0:2, :] # For 4D input data
                src = input_data
                tgt = target_data
                if src.shape[0] == 1:
                    src = src.squeeze(0)  # Remove batch dimension
                    tgt = tgt.squeeze(0)  # Remove batch dimension
    
                src = flat_over_img_caps(src)
                generated_sequences = torch.full((tgt.size(0), 1, 1), start_token_id, dtype=torch.long)
                running_val_loss = 0.0
                for k in range(tgt.size(2)):
                    decoder_input = tgt[:, :, 0:k+1]
                    output = model(src.to(my_device), decoder_input.to(my_device))
                    val_loss = compute_loss(output.contiguous().view(-1, vocabulary_size_target), decoder_input.squeeze(0).contiguous().view(-1).to(my_device))
                    print(f"Epoch: {epoch+1}, Iteration: {it+1}/{len(final_val_dataloader)}, Generation step: {k+1}/{tgt.size(2)}, Loss: {val_loss.item():.5f}")
                    running_val_loss += val_loss.item()
                    if k%10 == 0:
                        torch.cuda.empty_cache()

                avg_running_val_loss = running_val_loss/tgt.size(2)
                print(f"Epoch: {epoch+1}, Iteration {it+1}/{len(final_val_dataloader)}, Loss: {avg_running_val_loss:.5f}")
                val_loss_per_iter.append(avg_running_val_loss)
                epoch_val_loss += avg_running_val_loss

        avg_epoch_val_loss = epoch_val_loss/len(final_val_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_val_loss:.5f}")
        #val_loss_per_epoch.append(avg_val_loss)
        val_loss_per_epoch.append(val_loss_per_iter)

    print()
    print("Training finished!")
    for i in range(num_epochs):
        plot_losses(train_loss_per_epoch[i], val_loss_per_epoch[i], i)
        
    torch.save(model.state_dict(),'/data/admitosstorage/Phase_2/trained_models/auto/model_bert_yingjin_final.pt')

if type_of_training == 'seq':
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=int(number_epochs)*len(final_train_dataloader))
    train_seq2seq(int(number_epochs), scheduler)
if type_of_training == 'auto':
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=int(number_epochs)*len(final_train_dataloader)*max(all_dim3))
    train_auto(int(number_epochs), scheduler)



