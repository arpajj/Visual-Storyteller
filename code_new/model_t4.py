import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class T4Transformer(nn.Module):
    def __init__(self, vocab_size_input, vocab_size_target, d_model, num_layers, num_heads, d_ff, dropout, pad_token, device):
        super(T4Transformer, self).__init__()
        self.encoder = Encoder(vocab_size_input, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(vocab_size_target, d_model, num_layers, num_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size_target)
        self.dropout = nn.Dropout(dropout)
        self.pad_token = pad_token
        self.device = device

    def create_padding_mask(self, seq):
        return (seq != self.pad_token).unsqueeze(1).to(self.device)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.uint8).to(self.device)
        return mask == 0

    def create_masks(self, src, tgt):
        enc_mask = self.create_padding_mask(src)
        dec_padding_mask = self.create_padding_mask(src)
        look_ahead_mask = self.create_look_ahead_mask(tgt.size(1))
        dec_tgt_padding_mask = self.create_padding_mask(tgt)
        combined_mask = torch.max(dec_tgt_padding_mask, look_ahead_mask)
        return enc_mask, combined_mask, dec_padding_mask
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
    def forward(self, src, tgt):
        enc_mask, combined_mask, _ = self.create_masks(src, tgt)
        enc_output = self.encode(src, enc_mask)
        dec_output = self.decode(tgt, enc_output, enc_mask, combined_mask)
        final_output = self.fc_out(dec_output)
        return final_output

    def generate(self, input_captions, tokenizer, target_stories=None, teacher_forcing=False, max_length=200):
        self.eval()
        input_ids = tokenizer.encode(input_captions, return_tensors='pt').to(self.device)

        if target_stories != None:
            target_ids = tokenizer.encode(target_stories, return_tensors='pt').to(self.device)
            max_length = target_ids.shape[-1]
            
        generated_sequence = torch.full((input_ids.size(0), 1), tokenizer.cls_token_id, dtype=torch.long).to(self.device)
        for k in range(max_length):
            print("On generation step: ", k)
            enc_mask = self.create_padding_mask(input_ids).to(self.device)
            look_ahead_mask = self.create_look_ahead_mask(generated_sequence.size(1)).to(self.device)
            combined_mask = torch.max(self.create_padding_mask(generated_sequence), look_ahead_mask)
            
            decoder_input = generated_sequence
            if teacher_forcing and target_stories != None:
                decoder_input = target_ids[:, 0:k+1]

            enc_out = self.encode(input_ids, enc_mask)
            dec_out = self.decode(decoder_input, enc_out, enc_mask, combined_mask)

            logits = self.fc_out(dec_out)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1, keepdim=True)

            generated_sequence = torch.cat([generated_sequence, next_token], dim=-1)
            if next_token == tokenizer.sep_token_id:
                break
        
        return tokenizer.decode(generated_sequence.squeeze().cpu(), skip_special_tokens=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model//num_heads
        self.d_v = d_model//num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask) # with mask 
        #attn_output = self.self_attn(x, x, x) # without mask 
        x = self.layer_norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout2(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, enc_mask, combined_mask):
        attn1 = self.self_attn(x, x, x, combined_mask)
        x = self.layer_norm1(x + self.dropout1(attn1))
        attn2 = self.enc_dec_attn(x, enc_output, enc_output, enc_mask) # with encoder mask 
        #attn2 = self.enc_dec_attn(x, enc_output, enc_output) # without encoder mask 
        x = self.layer_norm2(x + self.dropout2(attn2))
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + self.dropout3(ff_output))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size_input, d_model, num_layers, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size_input, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size_target, d_model, num_layers, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size_target, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, enc_output, enc_mask, combined_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, enc_mask, combined_mask)
        return x










