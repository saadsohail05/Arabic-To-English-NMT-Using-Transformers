import torch
import torch.nn as nn
import numpy as np
import math

def get_device(use_cpu=False):
    """
    Determines the device to use for model computations.
    Args:
        use_cpu (bool): If True, forces CPU usage even if GPU is available.
    Returns:
        torch.device: Device to use for computations
    """
    if use_cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default to CPU for deployment safety
DEVICE = get_device(use_cpu=True)
PAD, UNK, BOS, EOS = 0, 1, 2, 3

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_input, key_input, value_input, mask=None):
        batch_size = query_input.size(0)
        query = self.query_proj(query_input).view(
            batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key_input).view(
            batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value_input).view(
            batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embedding_dim)
        return self.output_proj(output)

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input_tensor: torch.Tensor, sublayer) -> torch.Tensor:
        normalized = self.layer_norm(input_tensor)
        sublayer_output = sublayer(normalized)
        dropped = self.dropout(sublayer_output)
        return input_tensor + dropped

def get_positional_encoding(max_len, d_emb):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_emb)[np.newaxis, :]
    angles = pos / np.power(10000, 2 * i / d_emb)
    positional_encoding = np.zeros((max_len, d_emb))
    positional_encoding[:, ::2] = np.sin(angles[:, ::2])
    positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return positional_encoding[np.newaxis, ...]

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadedAttention(
            num_heads=config.num_attention_heads,
            embedding_dim=config.d_embed,
            dropout=config.dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_embed, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_embed)
        )
        self.skip1 = ResidualConnection(config.d_embed, config.dropout)
        self.skip2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, inputs, mask=None):
        attended = self.skip1(inputs, lambda x: self.attention(x, x, x, mask=mask))
        output = self.skip2(attended, self.ffn)
        return output

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.d_embed
        self.word_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        pos_encoding = get_positional_encoding(config.max_seq_len, config.d_embed)
        self.register_buffer('pos_encoding', torch.FloatTensor(pos_encoding))
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, tokens, mask=None):
        word_vectors = self.word_embed(tokens)
        pos_vectors = self.pos_encoding[:, :word_vectors.size(1), :]
        combined = self.dropout(word_vectors + pos_vectors)
        for block in self.blocks:
            combined = block(combined, mask)
        return self.norm(combined)

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.masked_self_attention = MultiHeadedAttention(config.num_attention_heads, config.d_embed)
        self.cross_attention = MultiHeadedAttention(config.num_attention_heads, config.d_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_embed)
        )
        self.residuals = nn.ModuleList([ResidualConnection(config.d_embed, config.dropout)
                                    for i in range(3)])

    def forward(self, encoder_output, encoder_mask, decoder_input, decoder_mask):
        decoder_state = self.residuals[0](decoder_input,
            lambda x: self.masked_self_attention(x, x, x, mask=decoder_mask))
        decoder_state = self.residuals[1](decoder_state,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, mask=encoder_mask))
        decoder_state = self.residuals[2](decoder_state, self.feed_forward)
        return decoder_state

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.d_embed
        self.token_embedding = nn.Embedding(config.decoder_vocab_size, config.d_embed)
        positional_encodings = get_positional_encoding(config.max_seq_len, config.d_embed)
        self.register_buffer('positional_encodings', torch.FloatTensor(positional_encodings))
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.decoder_layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.N_decoder)])
        self.layer_norm = nn.LayerNorm(config.d_embed)
        self.output_projection = nn.Linear(config.d_embed, config.decoder_vocab_size)

    def future_mask(self, sequence_length):
        causal_mask = (torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)!=0)
        device = next(self.parameters()).device
        return causal_mask.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, encoder_output, encoder_mask, target_tokens, target_padding_mask):
        sequence_length = target_tokens.size(1)
        token_embeddings = self.token_embedding(target_tokens)
        position_encoded = token_embeddings + self.positional_encodings[:, :sequence_length, :]
        decoder_state = self.embedding_dropout(position_encoded)
        future_mask = self.future_mask(sequence_length)
        future_mask = future_mask.expand(target_tokens.size(0), -1, sequence_length, sequence_length)
        if target_padding_mask.size(-1) != sequence_length:
            target_padding_mask = (target_tokens == 0).unsqueeze(1).unsqueeze(2)
        target_padding_mask = target_padding_mask.expand(-1, -1, sequence_length, sequence_length)
        attention_mask = target_padding_mask | future_mask
        for decoder_layer in self.decoder_layers:
            decoder_state = decoder_layer(encoder_output, encoder_mask, decoder_state, attention_mask)
        normalized_output = self.layer_norm(decoder_state)
        logits = self.output_projection(normalized_output)
        return logits

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_tokens, source_mask, target_tokens, target_mask):
        encoder_output = self.encoder(source_tokens, source_mask)
        return self.decoder(encoder_output, source_mask, target_tokens, target_mask)

class Config:
    def __init__(self):
        self.d_embed = 512
        self.feedforward_dim = 256
        self.num_attention_heads = 8
        self.N_encoder = 6
        self.N_decoder = 6
        self.dropout = 0.1
        self.max_seq_len = 512
        self.encoder_vocab_size = 30000
        self.decoder_vocab_size = 30000

def initialize_transformer(device=None, d_embed=512, feedforward_dim=2048, 
                         num_attention_heads=8, n_encoder=6, n_decoder=6, 
                         dropout=0.1, max_seq_len=512, 
                         encoder_vocab_size=30000, decoder_vocab_size=30000):
    if device is None:
        device = DEVICE
    config = Config()
    config.d_embed = d_embed
    config.feedforward_dim = feedforward_dim
    config.num_attention_heads = num_attention_heads
    config.N_encoder = n_encoder
    config.N_decoder = n_decoder
    config.dropout = dropout
    config.max_seq_len = max_seq_len
    config.encoder_vocab_size = encoder_vocab_size
    config.decoder_vocab_size = decoder_vocab_size
    
    encoder = Encoder(config)
    decoder = Decoder(config)
    model = Transformer(encoder, decoder)
    return model.to(device)

def translate(model, x):
    with torch.no_grad():
        dB = x.size(0)
        y = torch.tensor([[BOS]*dB]).view(dB, 1).to(DEVICE)
        x_pad_mask = (x == PAD).view(x.size(0), 1, 1, x.size(-1)).to(DEVICE)
        memory = model.encoder(x, x_pad_mask)
        for i in range(200):  # Using a reasonable max sequence length
            y_pad_mask = (y == PAD).view(y.size(0), 1, 1, y.size(-1)).to(DEVICE)
            logits = model.decoder(memory, x_pad_mask, y, y_pad_mask)
            last_output = logits.argmax(-1)[:, -1]
            last_output = last_output.view(dB, 1)
            y = torch.cat((y, last_output), 1).to(DEVICE)
    return y

def decode_sentence(detokenizer, sentence_ids):
    if not isinstance(sentence_ids, list):
        sentence_ids = sentence_ids.tolist()
    # Remove padding and eos tokens
    if EOS in sentence_ids:
        sentence_ids = sentence_ids[:sentence_ids.index(EOS)+1]
    while sentence_ids and sentence_ids[-1] == PAD:
        sentence_ids = sentence_ids[:-1]
    return detokenizer(sentence_ids).replace("<bos>", "").replace("<eos>", "").strip().replace(" .", ".")