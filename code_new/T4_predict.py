""" ########################### HERE ITS T4_PREDICT ################################"""
import pickle, torch, random
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from t4 import T4Transformer, T4tokenizer
from transformers import BertTokenizer

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
b_s = 8 # set the batch size

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

if name.lower()=='admitos' and use_bert:
    embedding_dim, number_layers, number_heads, feed_forward_dim = 768, 6, 12, 2048
    y = str(2)

if name.lower()=='admitos' and not use_bert:
    embedding_dim, number_layers, number_heads, feed_forward_dim = 512, 6, 8, 2048
    y = str(3)

if name.lower()=='yingjin' and use_bert:
    embedding_dim, number_layers, number_heads, feed_forward_dim = 512, 6, 8, 2048
    y = str(1)

if name.lower()=='yingjin' and not use_bert:
    embedding_dim, number_layers, number_heads, feed_forward_dim = 512, 6, 8, 2048
    y = str(1)

my_device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"
print("DEVICE USED: ", my_device)
model = T4Transformer(vocab_size_input=vocabulary_size_input, vocab_size_target=vocabulary_size_target, d_model=embedding_dim, num_layers=number_layers, 
                        num_heads=number_heads, d_ff=feed_forward_dim, dropout=0.2, pad_token=0, device=my_device, name=name)

model.load_state_dict(torch.load(f'/data/admitosstorage/Phase_2/trained_models/seq/model_bert_{name}_5x{y}.pt',map_location=my_device))
model = model.to(my_device)
model.eval()
print("The model has been loaded succesfully!!!")
print()

##### START THE EVALUATION ON THE TEST SET #####

print("-------------------------- Start Generating Stories --------------------------")

all_final_stories = []
with torch.no_grad():  # Disable gradient
    for it, (input_data, target_data) in enumerate(final_val_dataloader):
        #src = input_data[:, :, 0:2, :] # For 4D input data
        src = input_data
        tgt = target_data
        if src.shape[0] == 1:
            src = src.squeeze(0)  # Remove batch dimension
            tgt = tgt.squeeze(0)  # Remove batch dimension

        src = flat_over_img_caps(src)
        print("Generation on batch:", it)
        output = model(src.to(my_device), tgt.to(my_device))
        generated_stories = model.generate_text(output.detach(), tokenizer)
        all_final_stories.append(generated_stories)
        if it>=9:
            break    


print("-------------------------- Finish Generating Stories --------------------------")
print()
print("EXAMPLE OF SOME STORIES:")
choice_batch = random.randint(0,len(all_final_stories))
if name.lower() == 'admitos':
    stories_in_paragraph = [elem[0] for elem in all_final_stories[choice_batch]]
    stories_in_sentences = [elem[1] for elem in all_final_stories[choice_batch]]
if name.lower() == 'yingjin':
   stories_in_paragraph = [elem for elem in all_final_stories[choice_batch]]

for i, story in enumerate(stories_in_paragraph):
    print(f"Story {i+1}: {story}")
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

if name.lower() == 'admitos':
    for i, story in enumerate(stories_in_sentences):
        print(f"Story {i+1}:")
        for j, sent in enumerate(story):
            print(f"    Sentence {j+1}: {sent}")
        print("------------------------------------------------------------------------------------------------------------------")










