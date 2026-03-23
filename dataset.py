"""
Laboratório 05 - Treinamento Fim-a-Fim do Transformer
Disciplina: Tópicos em Inteligência Artificial
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuração
TOKENIZER_NAME = "bert-base-multilingual-cased"
MAX_LEN        = 32   
SUBSET_SIZE    = 1_000
BATCH_SIZE     = 16
PAD_IDX        = 0


# Carregamento do Dataset
def load_translation_data(subset_size: int = SUBSET_SIZE):
    print(f"Carregando dataset bentrevett/multi30k...")
    dataset = load_dataset("bentrevett/multi30k", split="train")

    subset      = dataset.select(range(subset_size))
    src_sentences = [ex["de"] for ex in subset]
    tgt_sentences = [ex["en"] for ex in subset]

    print(f"  {len(src_sentences)} pares carregados.")
    print(f"  Exemplo src: '{src_sentences[0]}'")
    print(f"  Exemplo tgt: '{tgt_sentences[0]}'")
    return src_sentences, tgt_sentences


# Tokenização
def build_tokenizer():
    print(f"\nCarregando tokenizador '{TOKENIZER_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  PAD id: {tokenizer.pad_token_id} | "
          f"CLS (START) id: {tokenizer.cls_token_id} | "
          f"SEP (EOS) id: {tokenizer.sep_token_id}")
    return tokenizer


def tokenize_pairs(src_sentences: list,
                   tgt_sentences: list,
                   tokenizer,
                   max_len: int = MAX_LEN):
    
    START_ID = tokenizer.cls_token_id
    EOS_ID   = tokenizer.sep_token_id
    PAD_ID   = tokenizer.pad_token_id

    def encode_and_pad(sentences, add_start=False, add_eos=False):
        all_ids = []
        for sent in sentences:
            ids = tokenizer.encode(sent,
                                   add_special_tokens=False,
                                   truncation=True,
                                   max_length=max_len - 2)
            if add_start:
                ids = [START_ID] + ids
            if add_eos:
                ids = ids + [EOS_ID]

            ids = ids[:max_len]
            ids += [PAD_ID] * (max_len - len(ids))
            all_ids.append(ids)
        return torch.tensor(all_ids, dtype=torch.long)

    src_ids    = encode_and_pad(src_sentences)
    tgt_input  = encode_and_pad(tgt_sentences, add_start=True)
    tgt_labels = encode_and_pad(tgt_sentences, add_eos=True)

    print(f"\nTokenização concluída:")
    print(f"  src_ids    shape: {src_ids.shape}")
    print(f"  tgt_input  shape: {tgt_input.shape}")
    print(f"  tgt_labels shape: {tgt_labels.shape}")
    return src_ids, tgt_input, tgt_labels


# Dataset PyTorch

class TranslationDataset(Dataset):

    def __init__(self, src_ids, tgt_input, tgt_labels):
        self.src_ids    = src_ids
        self.tgt_input  = tgt_input
        self.tgt_labels = tgt_labels

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return (self.src_ids[idx],
                self.tgt_input[idx],
                self.tgt_labels[idx])


def build_dataloader(src_ids, tgt_input, tgt_labels,
                     batch_size: int = BATCH_SIZE,
                     shuffle: bool = True) -> DataLoader:
    dataset = TranslationDataset(src_ids, tgt_input, tgt_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Execução Standalone para teste

if __name__ == "__main__":
    src, tgt         = load_translation_data()
    tokenizer        = build_tokenizer()
    src_ids, tgt_in, tgt_lbl = tokenize_pairs(src, tgt, tokenizer)
    loader           = build_dataloader(src_ids, tgt_in, tgt_lbl)

    # Mostra um batch de exemplo
    src_b, tgt_b, lbl_b = next(iter(loader))
    print(f"\nBatch de exemplo:")
    print(f"  src_batch  shape: {src_b.shape}")
    print(f"  tgt_batch  shape: {tgt_b.shape}")
    print(f"  label_batch shape: {lbl_b.shape}")
    print("\n[Tarefas 1 e 2 concluídas ✓]")
