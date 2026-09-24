"""Expand diffusion beams until ten distinct unseen catalog movies exist."""
import math
import time
import torch


def collect_candidates(engine, batch, state, history, minimum=10, maximum_beams=512):
    model = engine.model
    device = batch['input_ids'].device
    intents = state['intents']
    pools = [dict() for _ in intents]
    merged = {}
    attempts = []
    total_seconds = 0.0
    old_beams, old_val = model.num_beams, model.val_num_beams
    width = min(max(engine.beams, minimum), maximum_beams)
    budgets = []
    while True:
        budgets.append(width)
        if width == maximum_beams:
            break
        width = min(width*2, maximum_beams)
    try:
        for width in budgets:
            model.num_beams = model.val_num_beams = width
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                predictions, scores = model.generate(batch, n_return_sequences=width, return_scores=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            total_seconds += time.perf_counter()-start
            invalid = collisions = repeats = watched = 0
            for slot, (tokens, values) in enumerate(zip(predictions.cpu().tolist(), scores.cpu().tolist())):
                seen = set()
                for sid, score in zip(tokens, values):
                    if not math.isfinite(float(score)):
                        continue
                    matches = engine.token2items.get(tuple(sid), [])
                    invalid += not bool(matches)
                    collisions += len(matches) > 1
                    for item in matches:
                        external = str(engine.id2item[item])
                        if external in history:
                            watched += 1
                            continue
                        if external in seen:
                            repeats += 1
                            continue
                        seen.add(external)
                        movie = engine.movie(external)
                        movie['violation'] = bool(set(movie['genres']) & set(state['excluded_genres']))
                        movie['score'] = float(score)+math.log(max(intents[slot]['probability'], 1e-8))
                        if external not in pools[slot] or movie['score'] > pools[slot][external]['score']:
                            pools[slot][external] = movie
                        if external not in merged or movie['score'] > merged[external]['score']:
                            merged[external] = movie
            attempts.append({'beams_per_intent': width, 'generated_sids': width*len(intents),
                             'invalid_sids': invalid, 'colliding_sids': collisions,
                             'repeated_movies': repeats, 'history_excluded': watched,
                             'unique_unseen_movies_so_far': len(merged)})
            if len(merged) >= minimum:
                break
    finally:
        model.num_beams, model.val_num_beams = old_beams, old_val
    if len(merged) < minimum:
        raise ValueError(f'扩大到每意图 {maximum_beams} beams 后仍只有 {len(merged)} 部有效未看电影。'
                         '本轮不展示不足10部的列表，请降低文本引导强度或放宽需求。')
    return {
        'movies': sorted(merged.values(), key=lambda x: x['score'], reverse=True)[:minimum],
        'branches': [{'intent': intent, 'movies': sorted(pool.values(), key=lambda x: x['score'], reverse=True)[:minimum]}
                     for intent, pool in zip(intents, pools)],
        'decode_seconds': total_seconds, 'generation_attempts': attempts,
        'invalid_sids': sum(a['invalid_sids'] for a in attempts),
        'colliding_sids': sum(a['colliding_sids'] for a in attempts),
    }
