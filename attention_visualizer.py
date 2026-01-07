import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models_ultra import AttentionLayer


def decode_review(sequence, word_index, index_from=3):
    reverse_word_index = {value: key for key, value in word_index.items()}
    tokens = []
    for idx in sequence:
        if idx == 0:
            token = "<PAD>"
        elif idx == 1:
            token = "<START>"
        elif idx == 2:
            token = "<UNK>"
        else:
            token = reverse_word_index.get(idx - index_from, "<UNK>")
        tokens.append(token)
    text = " ".join([t for t in tokens if t != "<PAD>"])
    return tokens, text


def extract_attention_weights(model, x_sample):
    if x_sample.ndim == 1:
        x_sample = np.expand_dims(x_sample, axis=0)

    outputs = model.predict(x_sample, verbose=0)
    weights = None

    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        weights = outputs[1]
    else:
        model(x_sample, training=False)
        for layer in model.layers:
            if isinstance(layer, AttentionLayer):
                weights = layer.get_last_attention_weights()
                break

    if weights is None:
        return None

    if tf.is_tensor(weights):
        weights = weights.numpy()

    if weights.ndim == 3:
        weights = weights[0, :, 0]
    elif weights.ndim == 2:
        weights = weights[0]

    return weights


def visualize_attention(tokens, weights, save_path, top_k=20):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    length = min(len(tokens), len(weights))
    tokens = tokens[:length]
    weights = np.asarray(weights[:length])

    keep_idx = np.array([i for i, t in enumerate(tokens) if t != "<PAD>"], dtype=np.int64)
    tokens = [tokens[i] for i in keep_idx.tolist()]
    weights = weights[keep_idx]

    if len(weights) == 0:
        return []

    weight_sum = float(np.sum(weights))
    if weight_sum > 0:
        weights = weights / weight_sum

    top_k = min(top_k, len(weights))
    top_indices = np.argsort(weights)[-top_k:][::-1]
    top_items = [(tokens[i], float(weights[i])) for i in top_indices]

    x = np.arange(len(weights))
    plt.figure(figsize=(max(10, len(weights) * 0.08), 4))
    plt.bar(x, weights, color="#1f77b4", alpha=0.85)
    plt.title("Attention Weights")
    plt.xlabel("Token Position")
    plt.ylabel("Weight")
    plt.grid(axis="y", alpha=0.3)

    for i in top_indices:
        plt.text(i, weights[i], tokens[i], rotation=60, ha='left', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return top_items
