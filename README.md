# Transformer — Treinamento Fim-a-Fim

**Disciplina:** Tópicos em Inteligência Artificial  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior

Laboratório final da Unidade I. Integra o Transformer Encoder-Decoder
(Labs 01–04) a um dataset real do Hugging Face e implementa o loop
completo de treinamento (Forward → Loss → Backward → Step).

---

## Estrutura do Projeto

```
transformer-training/
├── dataset.py   # Tarefas 1 e 2 — Dataset, tokenização e padding
├── train.py     # Tarefas 3 e 4 — Modelo PyTorch, training loop e overfit test
└── README.md
```

---

## Pré-requisitos

```bash
pip install torch datasets transformers
```

---

## Como Rodar

```bash
# Roda pipeline completo: carrega dados, treina e testa overfitting
python train.py

# Só testa carregamento e tokenização
python dataset.py
```



```python
!pip install torch datasets transformers
!python train.py
```

---

## Pipeline de Treinamento

```
multi30k (1000 pares de-en)
        │
   Tokenização BERT multilingual
   + padding MAX_LEN=32
   + <START>/<EOS> no Decoder
        │
  ┌─────▼──────────────────────┐
  │  EncoderBlock × 2          │
  │  Self-Attention            │
  │  Add & Norm / FFN          │
  └──────────┬─────────────────┘
             │  Z
  ┌──────────▼─────────────────┐
  │  DecoderBlock × 2          │
  │  Masked Self-Attention     │
  │  Cross-Attention (Z)       │
  │  Add & Norm / FFN          │
  │  Linear → logits           │
  └──────────┬─────────────────┘
             │
     CrossEntropyLoss (ignore_index=PAD)
             │
          Adam optimizer
          loss.backward()
          optimizer.step()
```

---

## Hiperparâmetros

| Parâmetro | Valor | Paper original |
|---|---|---|
| `d_model` | 128 | 512 |
| `d_k / d_v` | 32 | 64 |
| `d_ff` | 256 | 2048 |
| `N layers` | 2 | 6 |
| `epochs` | 15 | — |
| `optimizer` | Adam | Adam |
| `dataset` | multi30k (1k) | WMT (4.5M) |


---

## Overfitting Test

Técnica clássica de debugging de redes neurais: treina o modelo sobre
apenas 2 frases fixas por 50 épocas. O modelo deve memorizar a tradução
exata, provando que a arquitetura assimila padrões matriciais com sucesso.

---

## Nota de Integridade Acadêmica

Partes complementadas com IA (Claude, Anthropic), revisadas por Heitor Viana.  

