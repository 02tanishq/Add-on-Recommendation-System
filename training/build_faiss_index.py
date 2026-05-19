import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from training.retrieval_dataset import RetrievalItemDataset
from models.two_tower import TwoTowerModel


# ── utility helpers ───────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── item embedding extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_item_embeddings(
    model:       TwoTowerModel,
    items_path:  str,
    text_embs_path: str,
    pid2idx_path:   str,
    batch_size:  int          = 64,
    num_workers: int          = 0,
    device:      torch.device = torch.device('cpu'),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes ALL items in the catalog through the trained item tower.

    Iterates the full item catalog in batches and collects
    L2-normalised item embeddings for FAISS indexing.

    Returns
    -------
    all_embeddings : (n_items, d_model)  float32 numpy array
    all_product_ids: (n_items,)          int64   numpy array
        product_id at position i corresponds to embedding at row i
    """
    model.eval()

    # ── item dataset ──────────────────────────────────────────────────────────
    item_ds = RetrievalItemDataset(
        items_path     = items_path,
        text_embs_path = text_embs_path,
        pid2idx_path   = pid2idx_path,
    )

    item_loader = DataLoader(
        item_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )

    print(f"  Encoding {len(item_ds):,} items "
          f"in {len(item_loader):,} batches...")

    all_embeddings  = []
    all_product_ids = []
    start           = time.time()

    for step, batch in enumerate(item_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Encode through item tower
        item_embs = model.encode_candidate(
            batch['text_emb']
        )  # (B, d_model)

        # L2 normalise — FAISS inner product = cosine sim for unit vectors
        item_embs = nn.functional.normalize(item_embs, dim=-1)

        all_embeddings.append(item_embs.cpu().numpy())
        all_product_ids.append(batch['product_id'].cpu().numpy())

        if (step + 1) % 20 == 0:
            print(f"    Encoded {(step+1)*batch_size:,} / "
                  f"{len(item_ds):,} items | "
                  f"Time {format_time(time.time()-start)}")

    all_embeddings  = np.vstack(all_embeddings).astype(np.float32)
    all_product_ids = np.concatenate(all_product_ids).astype(np.int64)

    print(f"  Extraction complete: {all_embeddings.shape}")
    return all_embeddings, all_product_ids


# ── FAISS index builder ───────────────────────────────────────────────────────

def build_faiss_index(
    embeddings:   np.ndarray,
    d_model:      int,
    index_type:   str = 'IVFFlat',
    n_centroids:  int = 100,
    n_probe:      int = 10,
    use_gpu:      bool = False,
) -> 'faiss.Index':
    """
    Builds a FAISS index over the item embeddings.

    Index types
    ───────────
    'Flat'     — exact search, no approximation
                 Best for: small catalogs (<10k items)
                 Speed: slow at scale

    'IVFFlat'  — Inverted File Index, approximate
                 Best for: medium catalogs (10k-1M items)
                 Speed: fast, good recall
                 Requires training on sample embeddings

    'IVFPQ'    — IVF + Product Quantisation, compressed
                 Best for: large catalogs (>1M items)
                 Speed: fastest, slightly lower recall

    For Instacart (~50k items) IVFFlat is the right choice.

    Parameters
    ----------
    embeddings  : (n_items, d_model) L2-normalised float32
    d_model     : embedding dimension
    index_type  : 'Flat' | 'IVFFlat' | 'IVFPQ'
    n_centroids : IVF cluster count  (rule of thumb: sqrt(n_items))
    n_probe     : clusters to search at query time  (higher=more recall)
    use_gpu     : move index to GPU for faster build (requires faiss-gpu)
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss not installed. Run:\n"
            "  pip install faiss-cpu   # CPU only\n"
            "  pip install faiss-gpu   # GPU support"
        )

    n_items = embeddings.shape[0]
    print(f"\n  Building FAISS index...")
    print(f"    Index type  : {index_type}")
    print(f"    n_items     : {n_items:,}")
    print(f"    d_model     : {d_model}")

    if index_type == 'Flat':
        # Exact search — inner product (= cosine for unit vectors)
        index = faiss.IndexFlatIP(d_model)

    elif index_type == 'IVFFlat':
        # Approximate — IVF with flat (non-compressed) storage
        quantizer = faiss.IndexFlatIP(d_model)
        index     = faiss.IndexIVFFlat(
            quantizer, d_model, n_centroids, faiss.METRIC_INNER_PRODUCT
        )
        index.nprobe = n_probe

        print(f"    Training IVF index on {n_items:,} vectors...")
        index.train(embeddings)
        print(f"    IVF training complete.")

    elif index_type == 'IVFPQ':
        # Approximate + compressed — IVF with Product Quantisation
        n_subvectors = min(d_model // 4, 32)  # must divide d_model
        quantizer    = faiss.IndexFlatIP(d_model)
        index        = faiss.IndexIVFPQ(
            quantizer, d_model,
            n_centroids, n_subvectors, 8,     # 8 bits per subvector
        )
        index.nprobe = n_probe
        print(f"    Training IVFPQ index...")
        index.train(embeddings)
        print(f"    IVFPQ training complete.")

    else:
        raise ValueError(
            f"index_type must be 'Flat', 'IVFFlat', or 'IVFPQ', "
            f"got {index_type}"
        )

    # Move to GPU if requested
    if use_gpu:
        try:
            res   = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"    Index moved to GPU.")
        except Exception as e:
            print(f"    GPU move failed ({e}), using CPU index.")

    # Add all item embeddings to the index
    print(f"    Adding {n_items:,} vectors to index...")
    index.add(embeddings)
    print(f"    Index contains {index.ntotal:,} vectors.")

    return index


# ── popularity prior builder ──────────────────────────────────────────────────

def build_popularity_prior(
    items_path:  str,
    pid2idx_path: str,
    output_path: str,
):
    """
    Builds a popularity prior dict:
        { product_id: normalised_popularity_score }

    Used by Stage 1 retrieval (Popular Item Retrieval branch)
    to supplement FAISS results with globally popular items.

    Saved as JSON to artifacts/indexes/popularity.json
    """
    import pandas as pd

    print("\n  Building popularity prior...")

    items = pd.read_parquet(items_path)

    with open(pid2idx_path) as f:
        pid2idx = {int(k): int(v) for k, v in json.load(f).items()}

    # Normalise popularity rank to [0, 1]
    if 'popularity_rank' in items.columns:
        max_pop = items['popularity_rank'].max()
        min_pop = items['popularity_rank'].min()
        items['pop_norm'] = (
            (items['popularity_rank'] - min_pop)
            / (max_pop - min_pop + 1e-6)
        )
    else:
        items['pop_norm'] = 0.5

    popularity = {
        int(row['product_id']): float(row['pop_norm'])
        for _, row in items.iterrows()
        if int(row['product_id']) in pid2idx
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(popularity, f)

    print(f"  Popularity prior saved → {output_path}  "
          f"({len(popularity):,} items)")


# ── index sanity check ────────────────────────────────────────────────────────

def sanity_check_index(
    index,
    embeddings:    np.ndarray,
    product_ids:   np.ndarray,
    n_test:        int = 5,
    k:             int = 10,
):
    """
    Quick sanity check — query the index with a few known embeddings
    and verify that the top-1 result is the query item itself.
    (Self-retrieval should always succeed for an exact/approximate index.)
    """
    import faiss

    print(f"\n  Sanity checking index with {n_test} queries...")

    test_idxs   = np.random.choice(len(embeddings), n_test, replace=False)
    test_vecs   = embeddings[test_idxs].astype(np.float32)
    test_pids   = product_ids[test_idxs]

    scores, indices = index.search(test_vecs, k)   # (n_test, k)

    hits = 0
    for i in range(n_test):
        top1_pos_in_catalog = indices[i, 0]
        retrieved_pid       = product_ids[top1_pos_in_catalog]
        expected_pid        = test_pids[i]
        match               = retrieved_pid == expected_pid
        hits                += int(match)
        status              = "✓" if match else "✗"
        print(f"    Query PID {expected_pid:6d} | "
              f"Top-1 PID {retrieved_pid:6d} | "
              f"Score {scores[i,0]:.4f} | {status}")

    print(f"  Self-retrieval accuracy: {hits}/{n_test}")


# ── save index and metadata ───────────────────────────────────────────────────

def save_index(
    index,
    product_ids:  np.ndarray,
    index_path:   str,
    meta_path:    str,
    d_model:      int,
    index_type:   str,
    n_items:      int,
):
    """
    Saves the FAISS index and its metadata to disk.

    Outputs
    -------
    index_path  → .faiss binary file
    meta_path   → .json with product_id array and config
    """
    import faiss

    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # If index is on GPU, move back to CPU before saving
    try:
        index = faiss.index_gpu_to_cpu(index)
    except Exception:
        pass

    faiss.write_index(index, index_path)
    print(f"  FAISS index saved → {index_path}")

    meta = {
        'd_model':     d_model,
        'index_type':  index_type,
        'n_items':     n_items,
        'product_ids': product_ids.tolist(),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    print(f"  Index metadata saved → {meta_path}")


# ── main build function ───────────────────────────────────────────────────────

def build_index(
    # ── paths ──────────────────────────────────────────────────────────────
    retrieval_checkpoint: str = "artifacts/checkpoints/retrieval_tower.pt",
    items_path:           str = "outputs/items_instacart.parquet",
    text_embs_path:       str = "outputs/text_embeddings_instacart.npy",
    pid2idx_path:         str = "outputs/pid2idx_instacart.json",
    index_path:           str = "artifacts/indexes/item_index.faiss",
    meta_path:            str = "artifacts/indexes/item_index_meta.json",
    popularity_path:      str = "artifacts/indexes/popularity.json",
    embeddings_save_path: str = "outputs/item_embeddings.npy",
    # ── model config ───────────────────────────────────────────────────────
    text_emb_dim:         int  = 384,
    d_model:              int  = 64,
    n_food_groups:        int  = 5,
    max_cart_len:         int  = 50,
    # ── index config ───────────────────────────────────────────────────────
    index_type:           str  = 'IVFFlat',
    n_centroids:          int  = 100,
    n_probe:              int  = 10,
    use_gpu:              bool = False,
    # ── extraction config ──────────────────────────────────────────────────
    batch_size:           int  = 64,
    num_workers:          int  = 0,
):
    """
    Full FAISS index building pipeline.

    Steps:
        1. Load trained retrieval tower checkpoint
        2. Encode all catalog items through item tower
        3. Build FAISS index over item embeddings
        4. Sanity check index with self-retrieval
        5. Save index + metadata + popularity prior
        6. Save raw item embeddings for ranker use

    Output artifacts:
        artifacts/indexes/item_index.faiss
        artifacts/indexes/item_index_meta.json
        artifacts/indexes/popularity.json
        outputs/item_embeddings.npy
    """
    start  = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  CartComplete — Build FAISS Index")
    print(f"  Device     : {device}")
    print(f"  Index type : {index_type}")
    print(f"  Checkpoint : {retrieval_checkpoint}")
    print(f"{'='*60}\n")

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading retrieval tower checkpoint...")

    with open(pid2idx_path) as f:
        pid2idx = json.load(f)
    n_items = len(pid2idx)

    model = TwoTowerModel(
        d_in=384,
        d_tower=128,
        d_out=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        temperature=0.07,
    ).to(device)

    ckpt = torch.load(retrieval_checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"  Checkpoint loaded (epoch {ckpt.get('epoch', '?')})")

    # ── extract item embeddings ───────────────────────────────────────────────
    print("\nExtracting item embeddings...")
    embeddings, product_ids = extract_item_embeddings(
        model          = model,
        items_path     = items_path,
        text_embs_path = text_embs_path,
        pid2idx_path   = pid2idx_path,
        batch_size     = batch_size,
        num_workers    = num_workers,
        device         = device,
    )

    # Save raw embeddings — used by ranker's item encoder at inference
    os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
    np.save(embeddings_save_path, embeddings)
    print(f"  Item embeddings saved → {embeddings_save_path}  "
          f"shape={embeddings.shape}")
    
    print("Embedding shape:", embeddings.shape)

    # ── build FAISS index ─────────────────────────────────────────────────────
    index = build_faiss_index(
        embeddings  = embeddings,
        d_model     = d_model,
        index_type  = index_type,
        n_centroids = n_centroids,
        n_probe     = n_probe,
        use_gpu     = use_gpu,
    )

    # ── sanity check ──────────────────────────────────────────────────────────
    sanity_check_index(
        index       = index,
        embeddings  = embeddings,
        product_ids = product_ids,
        n_test      = 5,
        k           = 10,
    )

    # ── save index ────────────────────────────────────────────────────────────
    print("\nSaving index and metadata...")
    save_index(
        index        = index,
        product_ids  = product_ids,
        index_path   = index_path,
        meta_path    = meta_path,
        d_model      = d_model,
        index_type   = index_type,
        n_items      = len(product_ids),
    )

    # ── build popularity prior ────────────────────────────────────────────────
    build_popularity_prior(
        items_path   = items_path,
        pid2idx_path = pid2idx_path,
        output_path  = popularity_path,
    )

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"  FAISS Index Build Complete")
    print(f"  Total time   : {format_time(total_time)}")
    print(f"  Index size   : {index.ntotal:,} vectors")
    print(f"\n  Output artifacts:")
    print(f"    {index_path}")
    print(f"    {meta_path}")
    print(f"    {popularity_path}")
    print(f"    {embeddings_save_path}")
    print(f"{'='*60}\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build FAISS index for CartComplete retrieval"
    )

    # Paths
    parser.add_argument('--retrieval_checkpoint',
                        default="artifacts/checkpoints/retrieval_tower.pt")
    parser.add_argument('--items',
                        default="outputs/items_instacart.parquet")
    parser.add_argument('--text_embs',
                        default="outputs/text_embeddings_instacart.npy")
    parser.add_argument('--pid2idx',
                        default="outputs/pid2idx_instacart.json")
    parser.add_argument('--index_path',
                        default="artifacts/indexes/item_index.faiss")
    parser.add_argument('--meta_path',
                        default="artifacts/indexes/item_index_meta.json")
    parser.add_argument('--popularity_path',
                        default="artifacts/indexes/popularity.json")
    parser.add_argument('--embeddings_save_path',
                        default="outputs/item_embeddings.npy")

    # Model
    parser.add_argument('--text_emb_dim',  type=int,  default=384)
    parser.add_argument('--d_model',       type=int,  default=64)
    parser.add_argument('--max_cart_len',  type=int,  default=50)

    # Index
    parser.add_argument('--index_type',   default='IVFFlat',
                        choices=['Flat', 'IVFFlat', 'IVFPQ'])
    parser.add_argument('--n_centroids',  type=int,  default=100)
    parser.add_argument('--n_probe',      type=int,  default=10)
    parser.add_argument('--use_gpu',      action='store_true')

    # Extraction
    parser.add_argument('--batch_size',   type=int,  default=512)
    parser.add_argument('--num_workers',  type=int,  default=4)

    args = parser.parse_args()

    build_index(
        retrieval_checkpoint = args.retrieval_checkpoint,
        items_path           = args.items,
        text_embs_path       = args.text_embs,
        pid2idx_path         = args.pid2idx,
        index_path           = args.index_path,
        meta_path            = args.meta_path,
        popularity_path      = args.popularity_path,
        embeddings_save_path = args.embeddings_save_path,
        text_emb_dim         = args.text_emb_dim,
        d_model              = args.d_model,
        max_cart_len         = args.max_cart_len,
        index_type           = args.index_type,
        n_centroids          = args.n_centroids,
        n_probe              = args.n_probe,
        use_gpu              = args.use_gpu,
        batch_size           = args.batch_size,
        num_workers          = args.num_workers,
    )