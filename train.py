
import torch
import torch.nn as nn
import torch.optim as optim
import math

# Hiperparâmetros
D_MODEL    = 128
D_K        = 32
D_V        = 32
D_FF       = 256
N_LAYERS   = 2
N_EPOCHS   = 15
LR         = 1e-3
PAD_IDX    = 0
MAX_LEN    = 32


# Blocos fundamentais
class ScaledDotProductAttention(nn.Module):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + mask) V"""

    def __init__(self, d_k: int):
        super().__init__()
        self.scale = math.sqrt(d_k)

    def forward(self, Q, K, V, mask=None):
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, V)


def causal_mask(seq_len: int, device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# Encoder Block
class EncoderBlock(nn.Module):
    """Self-Attention → Add&Norm → FFN → Add&Norm"""

    def __init__(self, d_model, d_k, d_v, d_ff):
        super().__init__()
        self.W_Q  = nn.Linear(d_model, d_k, bias=False)
        self.W_K  = nn.Linear(d_model, d_k, bias=False)
        self.W_V  = nn.Linear(d_model, d_v, bias=False)
        self.W_O  = nn.Linear(d_v, d_model, bias=False)
        self.attn = ScaledDotProductAttention(d_k)
        self.ffn  = FeedForward(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-Attention (sem máscara)
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        att = self.W_O(self.attn(Q, K, V))
        x   = self.ln1(x + att)
        # FFN
        x   = self.ln2(x + self.ffn(x))
        return x


# Decoder Block

class DecoderBlock(nn.Module):
    """Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm"""

    def __init__(self, d_model, d_k, d_v, d_ff):
        super().__init__()
        # Masked self-attention
        self.W_Q1  = nn.Linear(d_model, d_k, bias=False)
        self.W_K1  = nn.Linear(d_model, d_k, bias=False)
        self.W_V1  = nn.Linear(d_model, d_v, bias=False)
        self.W_O1  = nn.Linear(d_v, d_model, bias=False)
        # Cross-attention
        self.W_Q2  = nn.Linear(d_model, d_k, bias=False)
        self.W_K2  = nn.Linear(d_model, d_k, bias=False)
        self.W_V2  = nn.Linear(d_model, d_v, bias=False)
        self.W_O2  = nn.Linear(d_v, d_model, bias=False)

        self.attn  = ScaledDotProductAttention(d_k)
        self.ffn   = FeedForward(d_model, d_ff)
        self.ln1   = nn.LayerNorm(d_model)
        self.ln2   = nn.LayerNorm(d_model)
        self.ln3   = nn.LayerNorm(d_model)

    def forward(self, y, Z):
        seq_len = y.size(1)
        mask    = causal_mask(seq_len, y.device)

        # Masked Self-Attention
        Q1, K1, V1 = self.W_Q1(y), self.W_K1(y), self.W_V1(y)
        att1 = self.W_O1(self.attn(Q1, K1, V1, mask=mask))
        y    = self.ln1(y + att1)

        # Cross-Attention
        Q2, K2, V2 = self.W_Q2(y), self.W_K2(Z), self.W_V2(Z)
        att2 = self.W_O2(self.attn(Q2, K2, V2))
        y    = self.ln2(y + att2)

        # FFN
        y    = self.ln3(y + self.ffn(y))
        return y


# Transformer Completo

class Transformer(nn.Module):
    
    def __init__(self, vocab_size, d_model=D_MODEL, d_k=D_K,
                 d_v=D_V, d_ff=D_FF, n_layers=N_LAYERS,
                 max_len=MAX_LEN):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)

        self.encoder = nn.ModuleList(
            [EncoderBlock(d_model, d_k, d_v, d_ff) for _ in range(n_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(d_model, d_k, d_v, d_ff) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x)
        return x   # Z

    def decode(self, tgt, Z):
        y = self.tgt_embed(tgt)
        for layer in self.decoder:
            y = layer(y, Z)
        return self.output_proj(y)

    def forward(self, src, tgt):
        Z      = self.encode(src)
        logits = self.decode(tgt, Z)
        return logits


# Training Loop
def train(model, dataloader, vocab_size,
          n_epochs=N_EPOCHS, lr=LR, device='cpu'):

    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*52}")
    print(f"TRAINING LOOP  |  {n_epochs} épocas  |  lr={lr}")
    print(f"{'='*52}")

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for src_b, tgt_in_b, tgt_lbl_b in dataloader:
            src_b      = src_b.to(device)
            tgt_in_b   = tgt_in_b.to(device)
            tgt_lbl_b  = tgt_lbl_b.to(device)

            # Forward pass
            logits = model(src_b, tgt_in_b)   # (B, seq, vocab)

            # CrossEntropy espera (B*seq, vocab) vs (B*seq,)
            B, S, V = logits.shape
            loss = criterion(logits.view(B * S, V),
                             tgt_lbl_b.view(B * S))

            # Backward + Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        history.append(avg_loss)
        print(f"  Época {epoch:02d}/{n_epochs}  |  Loss: {avg_loss:.4f}")

    print(f"\n  Loss inicial : {history[0]:.4f}")
    print(f"  Loss final   : {history[-1]:.4f}")
    queda = (history[0] - history[-1]) / history[0] * 100
    print(f"  Queda total  : {queda:.1f}%")
    return history


# Overfitting Test
def overfit_test(model, tokenizer, device='cpu'):

    from dataset import tokenize_pairs, build_dataloader

    print(f"\n{'='*52}")
    print("OVERFITTING TEST (2 frases, 50 épocas)")
    print(f"{'='*52}")

    src_mini = ["Ein Mann geht.", "Eine Frau läuft."]
    tgt_mini = ["A man walks.", "A woman runs."]

    src_ids, tgt_in, tgt_lbl = tokenize_pairs(src_mini, tgt_mini, tokenizer)
    loader = build_dataloader(src_ids, tgt_in, tgt_lbl,
                              batch_size=2, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 51):
        model.train()
        for src_b, tgt_in_b, tgt_lbl_b in loader:
            src_b, tgt_in_b, tgt_lbl_b = (
                src_b.to(device), tgt_in_b.to(device), tgt_lbl_b.to(device)
            )
            logits = model(src_b, tgt_in_b)
            B, S, V = logits.shape
            loss = criterion(logits.view(B*S, V), tgt_lbl_b.view(B*S))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"  Época {epoch:02d}/50  |  Loss: {loss.item():.4f}")

    # Gera a tradução da 1ª frase do mini-conjunto
    print(f"\n  Frase fonte : '{src_mini[0]}'")
    print(f"  Esperado    : '{tgt_mini[0]}'")
    model.eval()
    with torch.no_grad():
        src_tensor = src_ids[0:1].to(device)
        Z          = model.encode(src_tensor)

        START_ID = tokenizer.cls_token_id
        EOS_ID   = tokenizer.sep_token_id
        generated = [START_ID]

        for _ in range(MAX_LEN):
            tgt_t  = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model.decode(tgt_t, Z)
            next_id = logits[0, -1, :].argmax().item()
            if next_id == EOS_ID:
                break
            generated.append(next_id)

        tokens = tokenizer.convert_ids_to_tokens(generated[1:])
        traducao = tokenizer.convert_tokens_to_string(tokens)
        print(f"  Gerado      : '{traducao}'")

    print("\n[Tarefas 3 e 4 concluídas ✓]")


# Execução Principal

if __name__ == "__main__":
    from dataset import (load_translation_data, build_tokenizer,
                         tokenize_pairs, build_dataloader)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}\n")

    # Dados
    src, tgt  = load_translation_data()
    tokenizer = build_tokenizer()
    src_ids, tgt_in, tgt_lbl = tokenize_pairs(src, tgt, tokenizer)
    loader    = build_dataloader(src_ids, tgt_in, tgt_lbl)

    # Modelo
    VOCAB_SIZE = tokenizer.vocab_size
    model     = Transformer(vocab_size=VOCAB_SIZE)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"\nModelo instanciado  |  parâmetros: {n_params:,}")

    # Training loop
    train(model, loader, VOCAB_SIZE, device=device)

    # Overfitting test
    overfit_test(model, tokenizer, device=device)
