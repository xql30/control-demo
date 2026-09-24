"""Inference-only free-text grounding; no target labels or output reranking."""
import numpy as np
import torch


def standardize(values):
    return (values - values.mean(dim=-1, keepdim=True)) / values.std(
        dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)


def project_sid_bias(codes, weights, codebook_size=64):
    """Convert a text-conditioned item distribution to SID log-density ratios."""
    codes = torch.as_tensor(codes, dtype=torch.long, device=weights.device)
    result = []
    for depth in range(codes.shape[1]):
        prior = torch.bincount(codes[:, depth], minlength=codebook_size).float() + .01
        posterior = torch.zeros_like(prior).scatter_add_(0, codes[:, depth], weights)
        posterior = posterior + .01 / len(codes)
        energy = (posterior / posterior.sum()).log() - (prior / prior.sum()).log()
        result.append((energy-energy.mean()).clamp(-5, 5))
    return torch.stack(result).cpu().numpy()


def free_text_bias(engine, positive_embeddings, negative_texts):
    items = engine.catalog_control_embeddings
    positives = torch.as_tensor(positive_embeddings, device=items.device)
    positive_scores = standardize(positives @ items.T)
    penalty = torch.zeros(len(items), device=items.device)
    if negative_texts:
        negatives = engine.encoder.encode(negative_texts, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)
        negative_scores = standardize(torch.as_tensor(negatives, device=items.device) @ items.T)
        penalty = negative_scores.amax(dim=0).clamp_min(0)
    biases, evidence = [], []
    for score in positive_scores:
        weights = torch.softmax(score-penalty, dim=0)
        biases.append(project_sid_bias(engine.local_codes, weights))
        indices = weights.topk(min(5, len(items))).indices.cpu().tolist()
        evidence.append([engine.movie(str(engine.id2item[i+1])) for i in indices])
    return np.stack(biases), evidence

