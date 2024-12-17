import torch
import torch.nn as nn
import math, re
import torch.nn.functional as F

class T4Transformer(nn.Module):
    def __init__(self, vocab_size_input, vocab_size_target, d_model, num_layers, num_heads, d_ff, dropout, pad_token, device, name=None):
        super(T4Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size_input, d_model)#.to("cuda:0")
        self.decoder_embedding = nn.Embedding(vocab_size_target, d_model)#.to("cuda:1")
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])#.to("cuda:0")
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])#.to("cuda:1")
        self.linear1 = nn.Linear(d_model, d_model//2)#.to("cuda:1")
        self.linear2 = nn.Linear(d_model//2, vocab_size_target)#.to("cuda:1")
        self.dropout = nn.Dropout(dropout)
        self.pad_token = pad_token
        self.device = device
        self.name = name

    def create_masks_2d(self, src, tgt):
        src_mask = (src != self.pad_token).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != self.pad_token).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(self.device)
        tgt_mask = tgt_mask.to(nopeak_mask.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def create_masks_3d(self, src, tgt):
        src_mask = (src != self.pad_token).unsqueeze(2).unsqueeze(3)
        tgt_mask = (tgt != self.pad_token).unsqueeze(2).unsqueeze(4)
        seq_length = tgt.size(-1)
        nopeak_mask = (1 - torch.triu(torch.ones(seq_length, seq_length), diagonal=1)).bool()
        # Expand the nopeak_mask to match the tgt_mask dimensions for broadcasting
        nopeak_mask = nopeak_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        nopeak_mask = nopeak_mask.to(self.device)
        tgt_mask = tgt_mask.to(nopeak_mask.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def encode(self, src, src_mask):
        for layer in self.encoder:
            src = layer(src, src_mask)
        return src
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        for layer in self.decoder:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        return tgt
    
    def forward(self, src, tgt):
        if len(src.shape)==2:
            src_mask, tgt_mask = self.create_masks_2d(src,tgt)
        if len(src.shape)==3:
            src_mask, tgt_mask = self.create_masks_3d(src,tgt)
        src_emb = self.encoder_embedding(src)
        src_emb = self.pos_encoding(src_emb)
        src_emb = src_emb.squeeze(0)
        memory = self.encode(src_emb, src_mask)
        tgt_emb = self.decoder_embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = tgt_emb.squeeze(0)
        dec_output = self.decode(tgt_emb, memory, src_mask, tgt_mask)
        output = self.dropout(F.relu(self.linear1(dec_output)))
        output = self.linear2(output)
        return output

    def generate(self, input_captions, target_stories, tokenizer):
        self.eval()  # Set the model to evaluation mode

        start_token_id = tokenizer.cls_token_id
        eos_token_id = tokenizer.sep_token_id
        input_captions = input_captions.to(self.device)
        target_stories = target_stories.to(self.device)
        max_length = target_stories.shape[-1]
        # Initialize the generated sequence with the start token
        generated_sequences = torch.full((input_captions.size(0), 1), start_token_id, dtype=torch.long)
        #generated_sequences = generated_sequences.to(self.device)

        for k in range(max_length):
            decoder_input = generated_sequences
            #decoder_input = target_stories[:, 0:k+1]
            print(decoder_input)

            # Forward pass through the model
            output = self(input_captions, decoder_input)
            print(output.shape)

            # Get logits for the last generated token
            next_token_logits = output[:, -1, :]  # Shape: (batch_size, vocab_size)
            probs = F.softmax(next_token_logits, dim=-1)

            # Get the most likely next token
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)  # Shape: (batch_size, 1)
            print(next_token)

            # Append the predicted token to the generated sequence
            generated_sequences = torch.cat((generated_sequences.to(next_token.device), next_token), dim=1)
            print(f"gen step: {k} -->", generated_sequences)

            # If EOS token is generated for all sequences, stop early
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        # Return the generated sequences without the start token
        return generated_sequences[:, 1:]

    
    def generate_text(self, logits, tokenizer, strategy='greedy'):
        if len(logits.shape)==3: 
            logits = logits.unsqueeze(0) # Make the output of the model 4-dimensional
        else: # Its already 4-dimensional 
            pass
        
        all_stories_in_btch = []
        for j in range(logits.shape[0]):
            generated_sentences = []
            for sentence_logits in logits[j,:,:,:]:
                sentence_tokens = []
    
                for token_logits in sentence_logits:
                    if strategy == 'greedy':
                        token_id = torch.argmax(F.softmax(token_logits, dim=-1)).item()
                    elif strategy == 'sampling':
                        probabilities = F.softmax(token_logits, dim=-1)
                        token_id = torch.multinomial(probabilities, 1).item()
                    else:
                        raise ValueError("Unsupported decoding strategy")
    
                    token_word = tokenizer.decode([token_id])
                    sentence_tokens.append(token_word)
    
                generated_sentence = ' '.join(sentence_tokens)
                generated_sentences.append(generated_sentence)
            all_stories_in_btch.append(generated_sentences)

        final_stories = [self.refine_text(story) for story in all_stories_in_btch]
        return final_stories

    def refine_text(self, sentences):
        tokens_to_remove = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
        merged_sentences = [sentence.split() for sentence in sentences]
        get_index_of_end_tok = [sentence.index('[SEP]') for sentence in merged_sentences]
        chopped_sentences = [[token for token in sentence[:idx+1]] for idx,sentence in zip(get_index_of_end_tok,merged_sentences)]
        refined_sentences =  [' '.join([token for token in sentence if token not in tokens_to_remove]) for sentence in chopped_sentences]
        story_sentences = [re.sub(r'\s([?.!,\'](?:\s|$))', r'\1', sentence) for sentence in refined_sentences]
        complete_story = ' '.join(story_sentences)
        if self.name == 'admitos':
            split_story = complete_story.split()
            merged_story = self.merge_subword_tokens(split_story)
            final_story = self.fix_apostrophe(' '.join(merged_story))
            merge_story_sent = []
            for sent in story_sentences:
                merged_sent = self.merge_subword_tokens(sent.split())
                final_sent_story = self.fix_apostrophe(' '.join(merged_sent))
                merge_story_sent.append(final_sent_story)
            return final_story, merge_story_sent
        if self.name == 'yingjin':
            split_story = complete_story.split()
            merged_story = self.merge_subword_tokens(split_story)
            final_story = self.fix_apostrophe(' '.join(merged_story))
            return final_story

    def merge_subword_tokens(self, tokens):
        merged_tokens = []
        for token in tokens:
            if token.startswith('##'):
                if merged_tokens:
                    merged_tokens[-1] += token[2:]
                else:
                    merged_tokens.append(token[2:])
            else:
                merged_tokens.append(token)
        return merged_tokens

    def fix_apostrophe(self, sentence):
        # Remove space before apostrophes
        sentence = re.sub(r"\s+'", r"'", sentence)
        # Remove space after apostrophes
        sentence = re.sub(r"'\s+", r"'", sentence)
        return sentence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)*(-(math.log(10000.0)/d_model)))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x_device = x.device
        seq_len = x.size(-2)
        pe = self.pe[:, :, :seq_len, :]
        x = x + pe.to(x_device)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        # print("---------------------------------------------------")
        # print("Attention Output: ", attn_output.shape)
        x = x + self.dropout(self.norm1(attn_output))
        x = x + self.dropout(self.norm2(self.linear2(F.relu(self.linear1(x)))))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self.norm1(attn_output))
        attn_output, _ = self.enc_attn(x, memory, memory, src_mask)
        if len(x.shape) == 3:
            x = x + self.dropout(self.norm2(attn_output[:x.shape[0],:,:]))
        elif len(x.shape)==4: 
            x = x + self.dropout(self.norm2(attn_output[:,:x.shape[1],:,:]))
        else: 
            raise ValueError("Unexpected tensor shape. The tensor must be either 3D or 4D.")
        x = x + self.dropout(self.norm3(self.linear2(F.relu(self.linear1(x)))))
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward_3d(self, query, key, value, mask=None):
        q_img_cap_dim, q_seq_len, _  = query.size()
        k_img_cap_dim, k_seq_len, _  = key.size()
        v_img_cap_dim, v_seq_len, _  = value.size()
        query = self.linear_q(query).view(q_img_cap_dim, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(k_img_cap_dim, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(v_img_cap_dim, v_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        if query.shape[0] != key.shape[0]:
            query_expanded = query.unsqueeze(1).expand(-1, key.shape[0], -1, -1, -1).reshape(query.shape[0]*key.shape[0], self.num_heads, q_seq_len, self.d_k)
            key_expanded = key.unsqueeze(0).expand(query.shape[0], -1, -1, -1, -1).reshape(query.shape[0]*key.shape[0], self.num_heads, k_seq_len, self.d_k)
            value_new = value.repeat(q_img_cap_dim,1,1,1)
            if mask is not None:
                mask_expanded = mask.repeat(q_img_cap_dim,1,1,1)
        else:
            query_expanded = query
            key_expanded = key
            value_new = value
            if mask is not None:
                mask_expanded = mask
        scores = torch.matmul(query_expanded, key_expanded.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask_expanded is not None:
            scores = scores.masked_fill(mask_expanded == 0, -1e12)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value_new)
        if query.shape[0] != key.shape[0]:
            context = context.transpose(1, 2).contiguous().view(q_img_cap_dim*k_img_cap_dim, q_seq_len, self.num_heads*self.d_k)
        else: 
            context = context.transpose(1, 2).contiguous().view(q_img_cap_dim, q_seq_len, self.num_heads*self.d_k)
        out = self.linear_out(context)

        return out, attn_weights

    def forward_4d(self, query, key, value, mask=None):
        btch_size, q_img_cap_dim, q_seq_len, _  = query.size()
        btch_size, k_img_cap_dim, k_seq_len, _  = key.size()
        btch_size, v_img_cap_dim, v_seq_len, _  = value.size()
        query = self.linear_q(query).view(btch_size, q_img_cap_dim, q_seq_len, self.num_heads, self.d_k).transpose(2, 3)
        key = self.linear_k(key).view(btch_size, k_img_cap_dim, k_seq_len, self.num_heads, self.d_k).transpose(2, 3)
        value = self.linear_v(value).view(btch_size, v_img_cap_dim, v_seq_len, self.num_heads, self.d_k).transpose(2, 3)
        if query.shape[1] != key.shape[1]:
            query_expanded = query.unsqueeze(2).expand(-1, -1, key.shape[1], -1, -1, -1).reshape(btch_size, query.shape[1]*key.shape[1], self.num_heads, q_seq_len, self.d_k)
            key_expanded = key.unsqueeze(1).expand(-1, query.shape[1], -1, -1, -1, -1).reshape(btch_size, query.shape[1]*key.shape[1], self.num_heads, k_seq_len, self.d_k)
            value_new = value.repeat(1,q_img_cap_dim,1,1,1)
            if mask is not None:
                mask_expanded = mask.repeat(1,q_img_cap_dim,1,1,1)
        else:
            query_expanded = query
            key_expanded = key
            value_new = value
            if mask is not None:
                mask_expanded = mask
        scores = torch.matmul(query_expanded, key_expanded.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask_expanded is not None:
            scores = scores.masked_fill(mask_expanded == 0, -1e12)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value_new)
        if query.shape[1] != key.shape[1]:
            context = context.transpose(2, 3).contiguous().view(btch_size, q_img_cap_dim*k_img_cap_dim, q_seq_len, self.num_heads*self.d_k)
        else: 
            context = context.transpose(2, 3).contiguous().view(btch_size, q_img_cap_dim, q_seq_len, self.num_heads*self.d_k)
        out = self.linear_out(context)

        return out, attn_weights

    def forward(self, query, key, value, mask=None):
        query_size = query.size()
        key_size = key.size()
        value_size = value.size()
        # print("Intial Query, Key, Value shapes", query.shape, "|", key.shape, "|", value.shape)
        if len(query_size)==3 or len(key_size)==3 or len(value_size)==3:
            output, attention_weights = self.forward_3d(query, key, value, mask)
        elif len(query_size)==4 or len(key_size)==4 or len(value_size)==4:
            output, attention_weights = self.forward_4d(query, key, value, mask)
        else:
            raise ValueError("Unexpected tensor shape. The tensor must be either 3D or 4D.")    

        return output, attention_weights


class T4tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}
        self.inv_vocab_dict = {idx: word for idx, word in enumerate(vocabulary)}
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.unk_token = '[UNK]'
        self.pad_token_id = self.vocab_dict[self.pad_token]
        self.cls_token_id = self.vocab_dict[self.cls_token] 
        self.sep_token_id = self.vocab_dict[self.sep_token]
        self.unk_token_id = self.vocab_dict[self.unk_token]
        self.special_tokens = [self.pad_token_id, self.cls_token_id, self.sep_token_id, self.unk_token_id]

    def tokenize(self, text):
        """Tokenizes a string into a list of tokens."""
        # Use regex to split the text into tokens and separate punctuation, including special tokens
        tokens = re.findall(r"\[.*?\]|[\w]+|[.,!?;]", text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a list of tokens to a list of token IDs."""
        return [self.vocab_dict.get(token, self.vocab_dict[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        """Converts a list of token IDs to a list of tokens."""
        return [self.inv_vocab_dict.get(idx, self.unk_token) for idx in ids]

    def pad_sequence(self, sequence, max_length):
        """Pads a list of token IDs to a specified length."""
        padding_length = max_length - len(sequence)
        if padding_length > 0: 
            return sequence + [self.vocab_dict[self.pad_token]]*padding_length
        else:
            return sequence[:max_length]
        
    def pad_batch(self, batch, max_length=None):
        """Pads a batch of token ID lists to the length of the longest list in the batch or to a specified max length."""
        if max_length is None:
            max_length = max(len(seq) for seq in batch)
        return [self.pad_sequence(seq, max_length) for seq in batch]

    def encode(self, text, return_tensors=None):
        """Encodes a string into a list of token IDs, adding special tokens."""
        tokens = [self.cls_token] + self.tokenize(text) + [self.sep_token]
        token_ids = self.convert_tokens_to_ids(tokens)
        if return_tensors == 'pt':
            return torch.tensor(token_ids).unsqueeze(0)
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Decodes a list of token IDs back into a string."""
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            text = ' '.join(tokens).replace(' [PAD]','').replace('[CLS]','').replace(' [SEP]','').replace(' .','.').replace(' ,',',').replace(' !','!').replace(' ?','?').replace(' ;',';')
        else: 
            text = ' '.join(tokens)
        return text

    def get_vocab(self):
        """Returns the Vocabulary in Dictionary form"""
        return self.vocab_dict
        