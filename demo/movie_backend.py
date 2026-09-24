"""Standalone checkpoint inference: no training datasets or local Qwen needed."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from demo.llm_api import ChatAPI

ROOT = Path(__file__).resolve().parents[1]

class MovieEngine:
    def __init__(self, assets=None, beams=30):
        from genrec.models.LLaDARec.model import LLaDARec
        self.root = Path(assets or ROOT/'assets').resolve()
        self.device = os.getenv('DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        self.llm = ChatAPI()
        self.catalog = json.loads((self.root/'catalog.json').read_text())
        self.id2item = ['[PAD]'] + sorted(self.catalog, key=int)
        self.item2id = {item:i for i,item in enumerate(self.id2item) if i}
        dataset = SimpleNamespace(n_items=len(self.id2item),item2id=self.item2id)
        raw_sids = json.loads((self.root/'item_sids.json').read_text())
        tokenizer = SimpleNamespace(n_digit=4,codebook_sizes=[64]*4,padding_token=0,
            eos_token=257,mask_token=258,vocab_size=259,ignored_label=-100,
            item2tokens={k:tuple(int(v[d])+1+d*64 for d in range(4)) for k,v in raw_sids.items()})
        config = {}
        for f in ['genrec/default.yaml','genrec/models/LLaDARec/config.yaml']:
            config.update(yaml.safe_load((ROOT/f).read_text()))
        config.update(device=torch.device(self.device),accelerator=SimpleNamespace(is_main_process=True),
            d_model=256,n_layers=4,n_heads=4,n_kv_heads=4,mlp_ratio=4,
            max_item_seq_len=20,gen_steps=4,num_beams=beams,val_num_beams=beams,temperature=1.,
            use_dialogue_multi_intent=True,dialogue_multi_intent_inject_mode='target_mask',
            dialogue_multi_intent_path=str(self.root/'initial_conditions.npz'),
            use_dialogue_multi_intent_sid_guidance=True,
            dialogue_multi_intent_sid_guidance_path=str(self.root/'initial_conditions.npz'),
            dialogue_multi_intent_sid_guidance_scale=3.)
        self.model = LLaDARec(config,dataset,tokenizer).to(self.device).eval()
        self.model.load_state_dict(torch.load(self.root/'recommender.pt',map_location='cpu',weights_only=True),strict=True)
        self.beams = beams
        self.model.num_beams = self.model.val_num_beams = beams
        self.genres = sorted({g for row in self.catalog.values() for g in row.get('genre',[])})
        self.tokens = self.model.item_id2tokens.detach().cpu().numpy()
        self.token2items = {}
        for i in range(1,len(self.tokens)):
            self.token2items.setdefault(tuple(self.tokens[i]),[]).append(i)
        self.local_codes = self.tokens[1:] - np.array([1,65,129,193])
        self.genre_sets = [set(self.catalog[item].get('genre',[])) for item in self.id2item[1:]]
        self.encoder = SentenceTransformer(str(self.root/'sentence-t5-base'),device=self.device)
        documents = []
        for item in self.id2item[1:]:
            row = self.catalog[item]
            fields = [row.get('name','')]
            for key in ['genre','director','writer','star']:
                value = row.get(key,[])
                fields.extend(value if isinstance(value,list) else [str(value)])
            fields.append(row.get('plot',''))
            documents.append('. '.join(str(v).strip() for v in fields if v))
        cache = self.root/'catalog_embeddings.npy'
        if cache.exists():
            vectors = np.load(cache, allow_pickle=False)
            if vectors.shape != (len(documents), 768):
                raise ValueError('Catalog embedding shape mismatch')
            self.catalog_control_embeddings = torch.as_tensor(vectors, device=self.device)
        else:
            self.catalog_control_embeddings = self.encoder.encode(documents,batch_size=64,
                normalize_embeddings=True,convert_to_tensor=True,show_progress_bar=True)
            np.save(cache, self.catalog_control_embeddings.cpu().numpy())
        # Hand-selected catalog examples, not histories of real users.
        self.profiles = {'示例：动作与剧情': [i for i in ['78418','99583','175096'] if i in self.item2id]}

    def movie(self, external):
        row = self.catalog.get(str(external), {})
        return {'id': str(external), 'title': row.get('name', str(external)),
                'genres': row.get('genre', [])}

    def parse(self, history, messages):
        from demo.preference_state import parse_preferences
        return parse_preferences(self, history, messages)

    def sid_bias(self, genres, banned):
        # Same catalog frequency grounding as the offline positive SID adapter;
        # negative feedback uses the allowed complement, without filtering output.
        if not genres and not banned:
            return np.zeros((4, 64), np.float32)
        mask = np.array([(not genres or bool(set(genres) & gs)) and not bool(set(banned) & gs)
                         for gs in self.genre_sets])
        if not mask.any():
            raise ValueError('当前类型条件在目录中无交集，请放宽条件。')
        codes = self.local_codes[mask]
        bias = []
        for d in range(4):
            counts = np.bincount(codes[:, d], minlength=64).astype(np.float32) + .01
            energy = np.log(counts / counts.sum())
            bias.append(np.clip(energy - energy.mean(), -5, 5))
        return np.asarray(bias)

    def recommend(self, history, state, strength=3.0):
        self.model.dialogue_multi_intent_sid_guidance_scale = float(strength)
        intents = state['intents']
        n = len(intents)
        emb = self.encoder.encode([i['text'] for i in intents], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        embeddings = torch.zeros((n, 4, emb.shape[1]), device=self.device)
        embeddings[:, :n] = torch.tensor(emb, device=self.device)
        probs = torch.zeros((n, 4), device=self.device)
        probs[torch.arange(n), torch.arange(n)] = 1
        bias = np.stack([self.sid_bias(i.get('genres', []), state['excluded_genres']) for i in intents])
        evidence = []
        if os.environ.get('MOVIE_FREE_TEXT', '1') != '0':
            from demo.semantic_control import free_text_bias
            semantic_bias, evidence = free_text_bias(self, emb, state['negative_intents'])
            bias = np.clip(bias + semantic_bias, -5, 5)
        biases = torch.zeros((n, 4, 4, 64), device=self.device)
        biases[:, :n] = torch.tensor(bias, device=self.device)
        self.model.dialogue_multi_intent_embeddings = embeddings
        self.model.dialogue_multi_intent_probabilities = probs
        self.model.dialogue_multi_intent_sid_bias = biases
        ids = [self.item2id[i] for i in history][-20:]
        batch = {'input_ids': torch.tensor([[0]*(20-len(ids))+ids]*n, device=self.device),
                 'user_ids': torch.arange(n, device=self.device), 'split': 'test'}
        torch.manual_seed(42)
        from demo.candidate_budget import collect_candidates
        result = collect_candidates(self, batch, state, history)
        result.update(semantic_grounding_examples=evidence, guidance_strength=float(strength),
                      free_text_control=os.environ.get('MOVIE_FREE_TEXT', '1') != '0')
        return result

    def chat(self, history, messages, strength=3.0, llm=None, preference_cache=None, language='zh'):
        if any(i not in self.item2id for i in history) or len(history) > 20:
            raise ValueError('历史最多 20 部电影，必须来自当前目录。')
        if not messages or len(messages) > 24 or any(len(m['content']) > 2000 for m in messages):
            raise ValueError('每条消息最多 2000 字，每次会话最多 12 轮，请重置后继续。')
        with self.lock:
            start = time.perf_counter()
            from demo.preference_state import parse_preferences
            context = SimpleNamespace(llm=llm or self.llm, genres=self.genres,
                catalog=self.catalog, movie=self.movie,
                _preference_cache=preference_cache if preference_cache is not None else {})
            state = parse_preferences(context, history, messages)
            if not math.isfinite(float(strength)) or not 0 <= float(strength) <= 8:
                raise ValueError('引导强度应在 0 到 8 之间。')
            result = self.recommend(history, state, strength)
            result['state'] = state
            from demo.recommendation_reply import explain_recommendation
            result.update(explain_recommendation(context, history, messages, result, language=language))
            return dict(result, total_seconds=time.perf_counter()-start)

    def history_only(self, history):
        """Generate from history with both conversational control paths disabled."""
        if any(i not in self.item2id for i in history) or len(history)>20:
            raise ValueError('Invalid viewing history')
        state={'intents':[{'text':'Movies matching the viewer interests','genres':[], 'probability':1.}],
               'negative_intents':[], 'excluded_genres':[], 'exclusions':{}, 'unresolved':[]}
        with self.lock:
            start=time.perf_counter()
            flags=(self.model.use_dialogue_multi_intent,self.model.use_dialogue_multi_intent_sid_guidance)
            try:
                self.model.use_dialogue_multi_intent=False
                self.model.use_dialogue_multi_intent_sid_guidance=False
                result=self.recommend(history,state,strength=0.)
            finally:
                self.model.use_dialogue_multi_intent,self.model.use_dialogue_multi_intent_sid_guidance=flags
            result.update(state=state,total_seconds=time.perf_counter()-start,history_only=True)
            return result
