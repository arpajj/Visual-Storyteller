# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md 
"""MMMMMMLLLLLLLLLLLLLLLLLPPPPPPPPPPPPPPPPPP """
import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys, pickle
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import skimage.io as io
import PIL.Image
#import cog

WEIGHTS_PATH1 = "/home/apassadaki/data/admitosstorage/pretrained_models/MLP_prefix/diiv2_prefix-003.pt" # path from-scratch 
#WEIGHTS_PATH1 = "/home/apassadaki/data/admitosstorage/pretrained_models/MSCOCO/MLP/dii_mlpO_vit-001.pt" # path from pre-trained in COCO  <----- 1st pick for ultimate phase 

WEIGHTS_PATH2 = "/home/apassadaki/data/admitosstorage/pretrained_models/MLP_prefix_GPT/diiv2_prefix-003.pt" # path from-scratch 
#WEIGHTS_PATH2 = "/home/apassadaki/data/admitosstorage/pretrained_models/MSCOCO/MLP/dii_mlpG_vit-001.pt" # path from pre-trained in COCO  <----- 2nd pick for ultimate phase 

IMAGES_PATH = "/home/apassadaki/data/admitosstorage/test_images/"

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
print(D)

class Predictor():

    def setup(self, weihts_path):
        """Load the model into memory to make running multiple predictions efficient"""
        
        self.device = D
        self.clip_model, self.preprocess = clip.load(ENCODER_TYPE, device=D, jit=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.prefix_length = 10 # fine-tuned on MSCOCO uses 10
        self.prefix_length = 10 # from-scratch uses 20 
        self.prefix_size = 640 if ENCODER_TYPE=="RN50x4" else 512
        print(ENCODER_TYPE, self.prefix_size)
        model = ClipCaptionModel(self.prefix_length, self.prefix_size)
        my_weights = torch.load(weihts_path, map_location=D)
        model.load_state_dict(my_weights)
        model = model.eval()
        model = model.to(D)

        return(model)

    # @cog.input("image", type=cog.Path, help="Input image")
    # @cog.input("model", type=str, options=WEIGHTS_PATHS.keys(), default="coco", help="Model to use")
    # @cog.input("use_beam_search", type=bool, default=False, help="Whether to apply beam search to generate the output text")
    
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
    def get_dummy_token(self, batch_size: int, device: D) -> torch.Tensor:
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


results_path1 = '/data/admitosstorage/results/fine_tuning/MLP/dii_prefix_only_vit_beam.pkl'
results_path2 = '/data/admitosstorage/results/fine_tuning/MLP/dii_prefix_GPT_vit_beam.pkl'

print("Results on path: ", results_path1)
predictor = Predictor()
my_model = predictor.setup(WEIGHTS_PATH1)
generated_caps = {}
print("The weights that will be used: ", WEIGHTS_PATH1)
for idx,filename in enumerate(os.listdir(IMAGES_PATH)):
    if filename.endswith(".jpg"):
        my_img_path = os.path.join(IMAGES_PATH,filename)
        if(int(filename[:-4])==39721865 or int(filename[:-4])==764437): # Truncated images for test set
            continue
        print("  Processing image: ", filename[:-4])
        # if int(filename[:-4])==1287553 or int(filename[:-4])==34380303 or int(filename[:-4])==642063 or int(filename[:-4])==114705299 or int(filename[:-4])==114842286 \
        # or  int(filename[:-4]) == 115077880 or int(filename[:-4])==158986134 or int(filename[:-4])==1086009868 or int(filename[:-4])==33125218:  # Truncated images for train set
        #     continue
        indiv_cap = predictor.predict(my_img_path, my_model, USE_BEAM_SEARCH)
        generated_caps[filename[:-4]] = indiv_cap
        #print("For the image {}, the caption is: {}".format(filename[:-4],indiv_cap))
        #if(idx>-1): break


# with open(results_path1, 'wb') as f:
#     pickle.dump(generated_caps, f)

# with open(results_path1, 'rb') as f:
#    my_res1 = pickle.load(f)

# print(len(my_res1))

# with open(results_path2, 'rb') as f:
#      my_res2 = pickle.load(f)

# print(len(my_res2))

