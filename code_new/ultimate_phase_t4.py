"""UUUULLLTIMATEEEEE PHASSEEEEE """
from transformers import BartForConditionalGeneration, BartTokenizer
import torch, pickle, json
import clip, os, re
from torch import nn
import numpy as np
import torch.nn.functional as nnf
import sys, pickle
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, BertTokenizer
from torch.optim import AdamW
import skimage.io as io
import PIL.Image, math
import torch.nn.functional as F

# mathcing_path_train = '/data/admitosstorage/Phase_2/data_phase2/final_strs_caps_match_train.json'
# mathcing_path_val = '/data/admitosstorage/Phase_2/data_phase2/final_strs_caps_match_val.json'
# mathcing_path_test = '/data/admitosstorage/Phase_2/data_phase2/final_strs_caps_match_test.json'

mathcing_path_train = 'C:/Users/admitos/Desktop/ThesisUU/Phase_2/data_phase2/final_strs_caps_match_train.json'
mathcing_path_val = 'C:/Users/admitos/Desktop/ThesisUU/Phase_2/data_phase2/final_strs_caps_match_val.json'
mathcing_path_test_new = 'C:/Users/admitos/Desktop/ThesisUU/Phase_2/data_phase2/final_strs_caps_match_test_new.json'

with open(mathcing_path_train, 'r', encoding='utf-8') as f:
    matching_dict_train = json.load(f)

with open(mathcing_path_val, 'r', encoding='utf-8') as f:
    matching_dict_val = json.load(f)

with open(mathcing_path_test_new, 'r', encoding='utf-8') as f:
    matching_dict_test = json.load(f)

# FROM THE SERVER
#WEIGHTS_PATH1 = "/home/apassadaki/data/admitosstorage/pretrained_models/MSCOCO/MLP/dii_mlpO_vit-001.pt" # path from pre-trained in COCO  <----- 1st pick for ultimate phase 
#WEIGHTS_PATH2 = "/home/apassadaki/data/admitosstorage/pretrained_models/MSCOCO/MLP/dii_mlpG_vit-001.pt" # path from pre-trained in COCO  <----- 2nd pick for ultimate phase 

# LOCALLY
#WEIGHTS_PATH1 = "C:/Users/admitos/Desktop/ThesisUU/pretrained_models/MSCOCO/MLP/dii_mlpO_vit-001.pt" # path from pre-trained in COCO  <----- 1st pick for ultimate phase 
WEIGHTS_PATH2 = "C:/Users/admitos/Desktop/ThesisUU/pretrained_models/MSCOCO/MLP/dii_mlpG_vit-001.pt" # path from pre-trained in COCO  <----- 2nd pick for ultimate phase 

#IMAGES_PATH = "/home/apassadaki/data/admitosstorage/test_images/"
IMAGES_PATH = "F:/ThesisUU/test/"


USE_BEAM_SEARCH = True
ENCODER_TYPE = "ViT-B/32"
#ENCODER_TYPE = "RN50x4"
is_gpu = True 
CPU = torch.device("cpu")

def get_device(device_id: int):
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')

CUDA = get_device 
D = CUDA(0) if is_gpu else "cpu"
print("Device used:", D)

class Predictor():

    def setup(self, weihts_path):
        """Load the model into memory to make running multiple predictions efficient"""
        
        self.device = D
        self.clip_model, self.preprocess = clip.load(ENCODER_TYPE, device=D, jit=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = 10 # fine-tuned on MSCOCO uses 10
        #self.prefix_length = 10 # from-scratch uses 20 
        self.prefix_size = 640 if ENCODER_TYPE=="RN50x4" else 512
        print("CLIP encoder:", ENCODER_TYPE, "with:",self.prefix_size)
        model = ClipCaptionModel(self.prefix_length, self.prefix_size)
        my_weights = torch.load(weihts_path, map_location=D)
        model.load_state_dict(my_weights)
        model = model.eval()
        model = model.to(D)
        return(model)
    
    def predict(self, image, model, use_beam_search):
        """Run a single prediction on the model"""
        image = io.imread(image)
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(D)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(D, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer, embed=prefix_embed)


class MLP(nn.Module):
    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 40:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size*prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size*prefix_length)//2, self.gpt_embedding_size*prefix_length))

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size, device):
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens, prefix, mask = None, labels = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None, entry_length=67, temperature=1.0, stop_token: str = ".",):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[: int(length)]) for output, length in zip(output_list, seq_lengths)]

    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1, entry_length=67,  # maximum number of words
                top_p=0.8,temperature=1.0, stop_token: str = "."):
    
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device
    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break
            if tokens.shape[1]>1:
                output_list = list(tokens.squeeze().cpu().numpy())
            else: 
                output_list = [tokens.squeeze().cpu().numpy()]
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

##################################### ------------------------- T4 architecture ------------------------ #####################################

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
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


def beam_search(model, input_sequence, SOS_token, EOS_token, beam_width, max_length=149):
        device = next(model.parameters()).device  # Get device from model parameters
    
        # Initialize sequences with the <cls> token (or <s> token)
        y_input = torch.tensor([[SOS_token]], dtype=torch.long).to(D)
        gen_seq = torch.tensor([[SOS_token]], dtype=torch.long).to(D)
    
        # Initialize beams with initial sequence
        beams = [(gen_seq, 0)]  # List of (sequence, cumulative log probability)
    
        src_mask = model.get_src_tgt_mask(input_sequence.size(1)).to(device)
    
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
        
        y_input = torch.tensor([[SOS_token]], dtype=torch.long).to(D)
        gen_seq = torch.tensor([[SOS_token]], dtype=torch.long).to(D)
        src_mask = model.get_src_tgt_mask(input_sequence.size(1)).to(D)
        
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
            tgt_mask = model.get_src_tgt_mask(y_input.size(1)).to(D)
            pred = model(input_sequence, y_input, src_mask, tgt_mask)
            
            if strategy == 'greedy':
                next_item = pred.topk(1)[1].view(-1)[-1].item()
            elif strategy == 'multinomial':
                next_item = torch.multinomial(F.softmax(pred[:, -1, :], dim=-1), 1).item()
            elif strategy == 'p-sampling':
                next_item = nucleus_sampling(pred[:, -1, :], p)
            else:
                raise ValueError("Unsupported decoding strategy")
            
            next_item = torch.tensor([[next_item]]).to(D)
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

# Load Vision-to-Caption model
predictor = Predictor()
my_clipcap_model = predictor.setup(WEIGHTS_PATH2)
print("1) Vision-to-Caption Model Loaded Succesfully!")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token_id = tokenizer.cls_token_id
eos_token_id = tokenizer.sep_token_id
model = Transformer(num_tokens=30522, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1, pad_token=0).to(D)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print("The trainable parameters of the model are: ", count_parameters(model))
model.load_state_dict(torch.load(f'C:/Users/admitos/Desktop/ThesisUU/Phase_2/trained_models/Pytorch_T4/united_model_pytorch_e25_v1.pt', map_location=D))

model.eval()
model = model.to(D)
print("2) Caption-to-Story Model Loaded Succesfully!")

all_generated_stories = {}
for i, (key,values) in enumerate(matching_dict_test.items()):
    if i == 0:
        break
    story_captions = []
    for img_path in values:
        indiv_cap = predictor.predict(img_path, my_clipcap_model, USE_BEAM_SEARCH)
        story_captions.append(indiv_cap) 
    
    input_text = ' [SEP] '.join(story_captions)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    target_ids = None 
    with torch.no_grad():
        generated_story_ids, my_story = predict(model, input_ids.to(D), target_ids, start_token_id, eos_token_id, teacher_forcing=False, strategy='beam_search')
    all_generated_stories[key] = my_story
    print(f"On story generation [{i+1}/{len(matching_dict_test)}]")
    
print()
# print("Input Captions: ", story_captions)
# print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
# print("Generated Story: \n", story)
# print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
# print("All generated storylines are:", len(all_generated_stories))
# results_path = f'/data/admitosstorage/Phase_2/results/ultimate_phase/storylines_ClipCap7_T4base4.pkl' 

# with open(results_path, 'wb') as f:
#     pickle.dump(all_generated_stories, f)




###################################           PERSONAL STORYLINE           #########################################
################### ------------------------ HERE GOES MY STORY ----------------------------- ######################

print("Here goes my story")
my_story_path = 'C:/Users/admitos/Desktop/ThesisUU/My_Visual_Story/'

story_captions = []
for filename in os.listdir(my_story_path): 
    if filename.endswith(".jpg") or filename.endswith(".png"):
        final_img_path = os.path.join(my_story_path, filename)
        indiv_cap = predictor.predict(final_img_path, my_clipcap_model, USE_BEAM_SEARCH)
        story_captions.append(indiv_cap) 

story_captions = ['A scenic view of Athens under the bright sun.', 'Tourists exploring the ancient ruins of the Acropolis.'
                "The Parthenon stands tall against the clear sky.", 'Photographing the sunset over Athens.', 'Dining with a view of the moonlit Acropolis']
input_text = ' [SEP] '.join(story_captions)
print(input_text)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
target_ids = None 
with torch.no_grad():
    generated_story_ids, my_story = predict(model, input_ids.to(D), target_ids, start_token_id, eos_token_id, teacher_forcing=False, strategy='beam_search')
all_generated_stories[key] = my_story
print()
print(my_story)
