from transformers import BartForConditionalGeneration, BartTokenizer
import torch, json
import clip
import os
from torch import nn
import numpy as np
import torch.nn.functional as nnf
from typing import Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel#
import skimage.io as io
import PIL.Image

mathcing_path_train = "path to the file 'final_strs_caps_match_train.json'. See example below"
# mathcing_path_train = 'C:/Users/admitos/Desktop/ThesisUU/Phase_2/data_phase2/final_strs_caps_match_train.json'
mathcing_path_val = "path to the 'final_strs_caps_match_val.json' file."
mathcing_path_test_new = "path to the 'final_strs_caps_match_test.json' file."

with open(mathcing_path_train, 'r', encoding='utf-8') as f:
    matching_dict_train = json.load(f)

with open(mathcing_path_val, 'r', encoding='utf-8') as f:
    matching_dict_val = json.load(f)

with open(mathcing_path_test_new, 'r', encoding='utf-8') as f:
    matching_dict_test = json.load(f)

WEIGHTS_PATH = "path to your captioner saved model it must be a '.pt' file. See example below"
# WEIGHTS_PATH = "C:/Users/admitos/Desktop/ThesisUU/pretrained_models/MSCOCO/MLP/dii_mlpG_vit-002.pt" 

IMAGES_PATH = "path to your test images"


USE_BEAM_SEARCH = False
## Encoder type of CLIP. It must me the same as the one that you used in vist_parser.py while getting the embeddings and then trained the captioner.
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


# Load Vision-to-Caption model
predictor = Predictor()
my_clipcap_model = predictor.setup(WEIGHTS_PATH2)
print("1) Vision-to-Caption Model Loaded Succesfully!")

# Load Caption-to-Story model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model.load_state_dict(torch.load(f"path to yout trainerd BART storyteller model. It must be a '.pt' file. See example below.", map_location=D))
#model.load_state_dict(torch.load(f'C:/Users/admitos/Desktop/ThesisUU/Phase_2/trained_models/BART/trained_bart_e3.pt', map_location=D))

model.eval()
model = model.to(D)
print("2) Caption-to-Story Model Loaded Succesfully!")

all_generated_stories = {}
for i, (key,values) in enumerate(matching_dict_test.items()):
    print("We process image with index:", i)
    story_captions = []
    for img_path in values:
        indiv_cap = predictor.predict(img_path, my_clipcap_model, USE_BEAM_SEARCH)
        story_captions.append(indiv_cap) 
    
    print(story_captions)
    input_text = ' </s> '.join(story_captions)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        summary_ids = model.generate(input_ids.to(D), max_length=200, num_beams=1, early_stopping=False, do_sample=True, top_p=0.9) ### nucleus sampling
        #summary_ids = model.generate(input_ids.to(D), max_length=200, num_beams=3, early_stopping=True) ### beam search
        story = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    all_generated_stories[key] = story
    print(story)
    print(f"On story generation [{i+1}/{len(matching_dict_test)}]")
    

#print("Input Captions: ", story_captions)
#print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
#print("Generated Story: \n", story)
#print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("All generated storylines are:", len(all_generated_stories))

results_path = f"path to a '.pkl' file where you want to store the final stories." 
with open(results_path, 'wb') as f:
    pickle.dump(all_generated_stories, f)
