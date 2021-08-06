import torch
import numpy as np
from transformers import *
from utils.config import *
import utils.sbert_wk.utils as ut

# Set model
config = AutoConfig.from_pretrained("binwang/bert-base-nli-stsb", cache_dir="./cache")
config.output_hidden_states = True
tokenizer = AutoTokenizer.from_pretrained(
    "binwang/bert-base-nli-stsb", cache_dir="./cache"
)
model = AutoModelWithLMHead.from_pretrained(
    "binwang/bert-base-nli-stsb", config=config, cache_dir="./cache"
)
model.cuda()

params = {
    "max_seq_length": 128,
    "layer_start": 4,
    "context_window_size": 2,
    "embed_method": "dissecting",
}

print(f"Total BERT Params {sum(np.prod(list(p.size())) for p in model.parameters())}")

### SIMILARITY FUNCTION ###
def get_similarity(sentences):
    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
    features_input_ids = []
    features_mask = []
    for sent_ids in sentences_index:
        # Truncate if too long
        if len(sent_ids) > params["max_seq_length"]:
            sent_ids = sent_ids[: params["max_seq_length"]]
        sent_mask = [1] * len(sent_ids)
        # Padding
        padding_length = params["max_seq_length"] - len(sent_ids)
        sent_ids += [0] * padding_length
        sent_mask += [0] * padding_length
        # Length Check
        assert len(sent_ids) == params["max_seq_length"]
        assert len(sent_mask) == params["max_seq_length"]

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)

    features_mask = np.array(features_mask)

    batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
    batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
    batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
    model.zero_grad()

    with torch.no_grad():
        features = model(**inputs)[1]

    # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to list
    # of (num_hidden_states, seq_len, hidden_dim) for each element in the batch.
    all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

    embed_method = ut.generate_embedding(params["embed_method"], features_mask)
    embedding = embed_method.embed(params, all_layer_embedding)

    return (
        embedding[0].dot(embedding[1])
        / np.linalg.norm(embedding[0])
        / np.linalg.norm(embedding[1])
    )


def similarity_wrapper(sentences):
    return [
        get_similarity([sentences[0], sentences[1]]),
        get_similarity([sentences[0], sentences[2]]),
    ]
