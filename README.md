# CartComplete — Basket-Aware Food Add-On Recommendation System

> *"Most add-on systems recommend what's popular. CartComplete recommends what's missing."*

A production-style, two-stage recommendation pipeline that reads the current cart as an **incomplete meal** and surfaces the items most likely to complete it — the same way a waiter would suggest a drink when you order food, or a dessert when the table looks unfinished.

Built on the **Instacart 2017** dataset. Inspired by [MTGR (Han et al., CIKM 2025)](https://arxiv.org/abs/2408.11608), [DIN (Zhou et al., KDD 2018)](https://arxiv.org/abs/1706.06978), [SASRec (Kang & McAuley, ICDM 2018)](https://arxiv.org/abs/1808.09781), and [S2SRec2 (Walmart, SIGIR 2025)](https://arxiv.org/abs/2507.09101).

---

## The core insight

Standard cross-sell systems score candidates based on **item-item similarity** — they recommend more of what you already have. CartComplete uses **complementarity** instead: given what is already in the cart, what food group is missing? Which item has the highest co-occurrence PMI with the current cart contents? What does the meal look like if this item is added?

This framing — *cart as incomplete meal, recommendation as gap-filling* — is the architectural principle that drives every component in the system.

---

## Architecture

```
User request: cart + context (hour, day, restaurant)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Retrieval                                    │
│  Two-Tower ANN (FAISS) + PMI Graph + Popularity priors  │
│  → ~100 candidate items                                 │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — Representation                               │
│  Set Transformer + SASRec → basket_repr                 │
│  User History Encoder     → user_repr                   │
│  Context Encoder          → context_repr                │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 — Cross Attention (novel contribution)         │
│  DIN-style attention: candidate queries cart items      │
│  Explicit cross features: PMI, category gap, price      │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4 — Gated Fusion + DCN-V2 Ranker                 │
│  Gated fusion learns per-signal importance              │
│  DCN-V2 learns high-order feature crosses               │
│  MLP: 512 → 256 → 128 → 64 → P(add item)              │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Training                                               │
│  L = L_BPR + 0.1 × L_coverage                          │
│  Coverage head predicts missing food groups (novel)     │
└─────────────────────────────────────────────────────────┘
```

### What makes this different from standard approaches

| Component | Standard approach | CartComplete |
|---|---|---|
| Cart encoding | Bag of item IDs, mean pooling | Set Transformer (permutation-invariant) + SASRec (sequential order) fused |
| Candidate scoring | Dot product with user embedding | DIN cross-attention: candidate queries each cart item |
| Feature interaction | Concatenation + MLP | DCN-V2 learns explicit high-order crosses (burger × night × no-drink) |
| Training signal | CTR binary cross-entropy | BPR pairwise ranking + category coverage auxiliary loss |
| Evaluation | AUC only | HR@5, HR@10, NDCG@10, MRR + **Category Coverage Rate** (novel metric) |

---

## Novel contributions

**1. Meal-aware cart encoding**
The Set Transformer branch treats the cart as an unordered set (permutation-invariant — the order you add items doesn't matter), while SASRec captures sequential intent. A CartFusionLayer combines both representations. This lets the model know both *what* is in the cart and *how* it was assembled.

**2. DIN cross-attention repurposed for complementarity**
Original DIN uses the target item to attend over user session history. We repurpose this: the candidate add-on item attends over the current cart items. Each cart item gets an attention weight answering "how much does this cart item signal that the candidate is needed?" The weights are saved and returned in the API response for explainability.

**3. Category coverage auxiliary loss**
A secondary head on the cart encoder predicts which food groups (main/side/drink/snack/dessert) are missing from the current cart. Trained with BCE at λ=0.1 alongside the main BPR ranking loss. This forces the cart encoder to learn meal structure, making the ranker aware of food-group completeness rather than just co-purchase probability.

**4. Category Coverage Rate evaluation metric**
Alongside standard HR@K and NDCG@K, we report Category Coverage Rate: of the top-5 recommendations, how many belong to distinct food groups? A system that recommends 5 drinks when the cart already has a drink scores low. A meal-aware system scores high. This metric is not reported in standard RecSys benchmarks and directly measures the quality of meal completion recommendations.

---

## Results

Ablation on Instacart 2017 test set (leave-one-out evaluation, full-catalog ranking):

| Model | HR@5 | HR@10 | NDCG@10 | MRR | Cat. Coverage |
|---|---|---|---|---|---|
| Popularity baseline | — | — | — | — | — |
| Item-KNN (co-occurrence) | — | — | — | — | — |
| SASRec only | — | — | — | — | — |
| + Set Transformer fusion | — | — | — | — | — |
| + DIN cross-attention | — | — | — | — | — |
| + Coverage loss (CartComplete) | — | — | — | — | — |

## Dataset

**Instacart 2017 Market Basket Analysis** — 3.4M orders, 206k users, 49k products.  
Download: `kaggle competitions download -c instacart-market-basket-analysis`

Key preprocessing decisions:
- Filter to food-only departments (removes personal care, household, pets)
- Map 21 Instacart departments → 5 food groups: main / side / drink / snack / dessert
- Generate synthetic prices from department-level priors (Instacart has no prices)
- Within-basket leave-one-out split: `add_to_cart_order` column gives exact add sequence
- Hard in-basket negatives: same department, not added — not random catalog items

---

## Project structure

```
cartcomplete/
├── data/instacart/              ← raw CSVs
├── preprocessing/               ← full pipeline: preprocess.py
├── models/
│   ├── item_encoder.py          ← ID + text + price + category → d_model
│   ├── set_transformer.py       ← permutation-invariant cart encoder
│   ├── user_encoder.py          ← mean-pool history + smart cold-start
│   ├── context_encoder.py       ← temporal + weather + festival + restaurant
│   ├── din_attention.py         ← DIN cross-attention for complementarity
│   ├── dcn_v2.py                ← explicit high-order feature crosses
│   ├── gated_fusion.py          ← per-signal importance gating
│   ├── ranker.py                ← coverage head + temperature scaling + BPR loss
│   ├── two_tower.py             ← retrieval: cart tower + candidate tower
│   └── addon_recsys.py          ← full end-to-end model
├── training/
│   ├── retrieval_dataset.py
│   ├── ranking_dataset.py
│   ├── collate.py
│   ├── losses.py                ← BPR + coverage loss
│   ├── train_retrieval.py
│   ├── train_ranker.py
│   ├── build_faiss_index.py
│   └── evaluate.py
├── inference/
│   └── recommend.py             ← AddOnRecommender class
├── artifacts/                   ← generated (not committed)
├── results/                     ← ablation table + metrics
├── notebooks/
│   ├── eda.ipynb
│   ├── ablation_study.ipynb
│   └── attention_visualization.ipynb
├── api/
│   ├── app.py                   ← FastAPI
│   └── routes.py
├── demo/
│   └── app.py                   ← Streamlit
├── configs/
│   ├── model.yaml
│   └── training.yaml
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cartcomplete.git
cd cartcomplete
pip install -r requirements.txt
```

**Requirements**
```
torch>=2.1.0
numpy
pandas
scipy
faiss-cpu          # or faiss-gpu
sentence-transformers
pyarrow
scikit-learn
fastapi
uvicorn
redis
streamlit
plotly
wandb              # optional, for experiment tracking
```

---

## Quickstart

### 1. Preprocess data

```bash
# Place Instacart CSVs in data/instacart/
python preprocessing/preprocess.py \
  --data_dir data/instacart \
  --output_dir artifacts/processed

# Generate MiniLM text embeddings (~3 min on CPU)
python preprocessing/preprocess.py \
  --data_dir data/instacart \
  --output_dir artifacts/processed \
  --run_embeddings
```

### 2. Train retrieval tower

```bash
python training/train_retrieval.py \
  --train_pairs artifacts/processed/train_pairs.parquet \
  --items artifacts/processed/items.parquet \
  --text_embs artifacts/embeddings/text_embeddings.npy \
  --pid2idx artifacts/mappings/pid2idx.json \
  --epochs 10 \
  --batch_size 512
```

### 3. Build FAISS index

```bash
python training/build_faiss_index.py \
  --retrieval_ckpt artifacts/checkpoints/retrieval_tower.pt \
  --items artifacts/processed/items.parquet \
  --text_embs artifacts/embeddings/text_embeddings.npy \
  --output_dir artifacts/indexes
```

### 4. Train ranker

```bash
python training/train_ranker.py \
  --train_pairs artifacts/processed/train_pairs.parquet \
  --val_pairs artifacts/processed/val_pairs.parquet \
  --items artifacts/processed/items.parquet \
  --text_embs artifacts/embeddings/text_embeddings.npy \
  --pid2idx artifacts/mappings/pid2idx.json \
  --pmi_path artifacts/graphs/pmi_matrix.npz \
  --retrieval_ckpt artifacts/checkpoints/retrieval_tower.pt \
  --d_model 64 \
  --epochs 8 \
  --batch_size 32
```

### 5. Evaluate

```bash
python training/evaluate.py \
  --model_checkpoint artifacts/checkpoints/full_model.pt \
  --retrieval_ckpt artifacts/checkpoints/retrieval_tower.pt \
  --test_pairs artifacts/processed/test_pairs.parquet \
  --index_path artifacts/indexes/item_index.faiss \
  --meta_path artifacts/indexes/item_index_meta.json
```

### 6. Run the demo

```bash
# Start API
uvicorn api.app:app --reload --port 8000

# In another terminal, start Streamlit demo
streamlit run demo/app.py
```

### 7. Single inference

```python
from inference.recommend import AddOnRecommender

recommender = AddOnRecommender.from_artifacts(
    artifacts_dir  = "artifacts",
    checkpoint     = "artifacts/checkpoints/full_model.pt",
    retrieval_ckpt = "artifacts/checkpoints/retrieval_tower.pt",
)

result = recommender.recommend(
    cart_product_ids = [196, 25133, 38928],   # Instacart product IDs
    top_k            = 5,
    hour             = 19,                    # 7pm dinner
    dow              = 4,                     # Friday
    meal_period      = 4,                     # dinner
    restaurant_id    = 0,
)

for rec in result.recommendations:
    print(f"{rec.rank}. {rec.product_name} — {rec.reason}")
    print(f"   PMI: {rec.pmi_score:.3f} | Score: {rec.score:.3f}")
```

---

## API

Start the server with `uvicorn api.app:app --reload`.

### `POST /recommend`

```json
{
  "cart_product_ids": [196, 25133, 38928],
  "top_k": 5,
  "hour": 19,
  "dow": 4,
  "meal_period": 4,
  "restaurant_id": 0
}
```

Response:
```json
{
  "recommendations": [
    {
      "product_id": 47209,
      "product_name": "Sparkling Water",
      "food_group": "drink",
      "price": 2.99,
      "score": 0.847,
      "rank": 1,
      "reason": "adds a drink to complete your meal",
      "attention_weights": {"196": 0.72, "25133": 0.21, "38928": 0.07}
    }
  ],
  "retrieval_time_ms": 4.2,
  "ranking_time_ms": 18.1,
  "total_time_ms": 22.3,
  "n_candidates": 98
}
```

### `GET /health`

```json
{"status": "ok", "model": "CartComplete v1.0"}
```

---

## The attention heatmap

Every recommendation comes with DIN attention weights — which cart items drove the recommendation. In the Streamlit demo, clicking any recommendation shows a heatmap:

```
              Biryani   Naan   Raita
              ───────   ────   ─────
Mango Lassi    0.72    0.21    0.07
Gulab Jamun    0.31    0.18    0.51
Papad          0.44    0.39    0.17
```

"Mango Lassi was primarily driven by the Biryani in your cart — the model learned that Biryani and Lassi are a common food pairing."

---

## Papers cited

This project builds directly on the following research:

| Paper | What we use |
|---|---|
| [MTGR — Han et al., CIKM 2025](https://arxiv.org/abs/2408.11608) | Cross features insight, gated fusion, Group-Layer Norm inspiration |
| [DIN — Zhou et al., KDD 2018](https://arxiv.org/abs/1706.06978) | Target-aware attention mechanism (repurposed for cart→candidate) |
| [SASRec — Kang & McAuley, ICDM 2018](https://arxiv.org/abs/1808.09781) | Causal self-attention sequential encoder |
| [Set Transformer — Lee et al., NeurIPS 2019](https://arxiv.org/abs/1810.00825) | Permutation-invariant basket encoder |
| [LightGCN — He et al., SIGIR 2020](https://arxiv.org/abs/2002.02126) | Graph convolution for item-item complementarity graph |
| [DCN-V2 — Wang et al., WWW 2021](https://arxiv.org/abs/2008.13535) | Explicit high-order feature interaction learning |
| [NPA — Ariannezhad et al., arXiv 2024](https://arxiv.org/abs/2401.16433) | Within-basket recommendation, multi-intent attention |
| [S2SRec2 — Walmart, SIGIR 2025](https://arxiv.org/abs/2507.09101) | Set-to-set basket completion, coverage loss design |
| [Wide & Deep — Cheng et al., DLRS 2016](https://arxiv.org/abs/1606.07792) | Memorization + generalization MLP design |

---

## Limitations

- **Short sequences**: Instacart users have ~7 orders on average. SASRec is designed for longer histories — performance would improve with richer user data.
- **No item prices**: Instacart does not include prices. Synthetic prices from department priors are used — real prices would strengthen the price-ratio cross features.
- **No real-time signals**: Stock availability, current promotions, and live restaurant inventory are not modeled.
- **Social graph absent**: Unlike the Yelp version of this system, Instacart has no friend graph — the LightGCN social layer is not used here.
- **Weather and festival signals**: The context encoder supports these but Instacart has no weather data — those branches receive zeros during training.

---

## Future work

- Multi-scenario foundation model: one model serving all surfaces (homepage, checkout, reorder)
- Semantic IDs (TIGER/LIGER style) for cold-start item representation
- DPO preference alignment post-training (OneRec-style)
- Chain-of-thought reasoning for interpretable recommendations
- Real food delivery dataset (Delivery Hero DHRD) as second benchmark

---

## Author

**Tanishq** · [GitHub: 02tanishq](https://github.com/02tanishq) · [HuggingFace: Yellow02](https://huggingface.co/Yellow02)

Built as a summer AI/ML project. Part of a personal challenge: one production-quality AI/ML project every 10 days.

---

## License

MIT License. See `LICENSE` for details.

Dataset license: Instacart dataset is provided under the [Instacart Online Grocery Shopping Dataset 2017](https://www.kaggle.com/competitions/instacart-market-basket-analysis) terms. Do not redistribute the raw data files.
