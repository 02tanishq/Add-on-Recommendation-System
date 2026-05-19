"""
=============================================================================
 CartComplete — Full Data Preprocessing Pipeline
=============================================================================
 Datasets:
   - Instacart 2017 Market Basket Analysis (primary)
   - Delivery Hero Recommendation Dataset / DHRD (secondary)

 What this script produces:
   outputs/
   ├── items_instacart.parquet              ← enriched item table (price, food group, text emb)
   ├── users_instacart.parquet              ← user profiles (exploration score, affinity)
   ├── train_pairs_instacart.parquet        ← (cart, positive, negatives) training pairs
   ├── val_pairs_instacart.parquet
   ├── test_pairs_instacart.parquet
   ├── pmi_matrix_instacart.npz             ← item-item co-occurrence (sparse)
   ├── text_embeddings_instacart.npy        ← MiniLM embeddings per item
   ├── id_maps.json               ← string ID → integer index maps
   └── dataset_stats.json         ← sizes, split counts, class balance

 Run order:
   pip install pandas numpy scipy sentence-transformers tqdm pyarrow
   python Data_processing.py --dataset instacart --data_dir ./data/instacart
   python Data_processing.py --dataset dhrd     --data_dir ./data/dhrd
   python Data_processing.py --dataset both     --data_dir ./data
=============================================================================
"""

import os, json, zipfile, argparse, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Output directory ────────────────────────────────────────────────────────
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Instacart department → food group mapping ────────────────────────────────
# Maps Instacart's 21 departments to the 5 food groups your coverage loss needs
DEPT_TO_FOODGROUP = {
    "meat seafood":        "main",
    "deli":                "main",
    "breakfast":           "main",
    "frozen":              "main",
    "bakery":              "side",
    "produce":             "side",
    "dairy eggs":          "side",
    "pantry":              "side",
    "canned goods":        "side",
    "dry goods pasta":     "side",
    "beverages":           "drink",
    "alcohol":             "drink",
    "snacks":              "snack",
    "candy":               "dessert",
    "babies":              None,       # non-food → filtered
    "personal care":       None,
    "household":           None,
    "pets":                None,
    "missing":             None,
    "international":       "side",
    "bulk":                "side",
    "other":               None,
}

# Synthetic price ranges per department (mean, std) in USD
# Seeded by product_id → reproducible per item
DEPT_PRICE_PARAMS = {
    "main":    (9.5,  2.5),
    "side":    (4.2,  1.2),
    "drink":   (3.1,  1.0),
    "snack":   (3.8,  1.1),
    "dessert": (5.0,  1.5),
}

# ─── Temporal context helpers ─────────────────────────────────────────────────
def hour_to_meal_period(hour):
    if   6  <= hour < 10: return "breakfast"
    elif 10 <= hour < 14: return "lunch"
    elif 14 <= hour < 17: return "afternoon"
    elif 17 <= hour < 22: return "dinner"
    else:                 return "late_night"

def day_to_type(dow):
    # Instacart: 0 = Saturday (most orders), consistent across dataset
    return "weekend" if dow in [0, 1] else "weekday"


# =============================================================================
# INSTACART PREPROCESSING
# =============================================================================

def load_instacart(data_dir):

    print("\n" + "="*60)
    print(" LOADING INSTACART FILES")
    print("="*60)

    aisles      = pd.read_csv(os.path.join(data_dir, "aisles.csv"))
    departments = pd.read_csv(os.path.join(data_dir, "departments.csv"))
    products    = pd.read_csv(os.path.join(data_dir, "products.csv"))
    orders      = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    prior       = pd.read_csv(os.path.join(data_dir, "order_products__prior.csv"))
    train_split = pd.read_csv(os.path.join(data_dir, "order_products__train.csv"))

    print(f"  orders: {len(orders):>10,}")
    print(f"  products: {len(products):>10,}")
    print(f"  prior interactions: {len(prior):>10,}")

    # =========================================================
    # DOWNSAMPLING
    # =========================================================

    print("\n── Downsampling to ~10M interactions...")

    all_users = orders["user_id"].drop_duplicates()

    sampled_users = all_users.sample(
        frac=0.24,
        random_state=42
    )

    orders = orders[orders["user_id"].isin(sampled_users)]

    valid_order_ids = set(orders["order_id"])

    prior = prior[prior["order_id"].isin(valid_order_ids)]

    train_split = train_split[
        train_split["order_id"].isin(valid_order_ids)
    ]

    print(f"  Remaining users: {len(sampled_users):,}")
    print(f"  Remaining interactions: {len(prior):,}")

    # =========================================================
# SAVE DOWNSAMPLED DATASET
# =========================================================

    os.makedirs("outputs/downsampled_instacart", exist_ok=True)

    orders.to_parquet(
        "outputs/downsampled_instacart/orders.parquet",
        index=False
    )

    prior.to_parquet(
        "outputs/downsampled_instacart/prior.parquet",
        index=False
    )

    train_split.to_parquet(
        "outputs/downsampled_instacart/train_split.parquet",
        index=False
    )

    products.to_parquet(
        "outputs/downsampled_instacart/products.parquet",
        index=False
    )

    print("\n── Saved downsampled dataset")

    return aisles, departments, products, orders, prior, train_split


def build_instacart_items(products, aisles, departments):
    """
    Build enriched item table with:
      - food_group (main/side/drink/snack/dessert/None)
      - synthetic price (deterministic, seeded per product)
      - department and aisle names
    """
    print("\n── Building item table...")

    # Merge all metadata
    items = products.merge(aisles,      on="aisle_id")       \
                    .merge(departments, on="department_id")

    # Normalize department names for mapping
    items["department"] = items["department"].str.lower().str.strip()

    # Assign food group
    items["food_group"] = items["department"].map(DEPT_TO_FOODGROUP)

    # Filter to food-only items (drop personal care, household, etc.)
    n_before = len(items)
    items = items[items["food_group"].notna()].reset_index(drop=True)
    print(f"  Items after food filter: {len(items):,} (dropped {n_before - len(items):,} non-food)")

    # Generate synthetic prices — seeded by product_id for reproducibility
    prices = []
    for _, row in items.iterrows():
        rng = np.random.default_rng(seed=int(row["product_id"]))
        fg  = row["food_group"]
        mean, std = DEPT_PRICE_PARAMS.get(fg, (4.0, 1.0))
        price = max(0.5, rng.normal(mean, std))   # floor at $0.50
        prices.append(round(price, 2))
    items["price"] = prices

    print(f"  Price range: ${items['price'].min():.2f} – ${items['price'].max():.2f}")
    print(f"  Food group distribution:\n{items['food_group'].value_counts().to_string()}")
    return items


def build_instacart_users(orders, prior, items):
    """
    Build user profile table with:
      - exploration_score  (how adventurous a shopper)
      - category_affinity  (which food groups they order most)
      - addon_history_rate (how often they add more than 1 item)
      - avg_basket_value   (synthetic, from item prices)
      - avg_basket_size
    """
    print("\n── Building user profiles...")

    food_pids = set(items["product_id"])

    # Filter prior to food items only
    prior_food = prior[prior["product_id"].isin(food_pids)].copy()
    prior_food = prior_food.merge(items[["product_id", "food_group", "price"]], on="product_id")
    prior_food = prior_food.merge(orders[["order_id", "user_id", "order_hour_of_day",
                                          "order_dow", "days_since_prior_order"]], on="order_id")

    # --- exploration score ---
    # = normalized std of food_group distribution per user
    def exploration(grp):
        vc   = grp["food_group"].value_counts(normalize=True)
        # entropy-like: higher entropy = more exploration
        return float(-(vc * np.log(vc + 1e-9)).sum())

    print("  Computing exploration scores...")
    exp_scores  = prior_food.groupby("user_id").apply(exploration)
    exp_norm    = (exp_scores - exp_scores.min()) / (exp_scores.max() - exp_scores.min() + 1e-9)
    exp_norm = exp_norm.rename("exploration_score")

    # --- category affinity (fraction of orders containing each food group) ---
    print("  Computing category affinity...")
    # for fg in ["main", "side", "drink", "snack", "dessert"]:
    #     grp_fg = prior_food[prior_food["food_group"] == fg].groupby("user_id")["order_id"].nunique()
    #     total  = prior_food.groupby("user_id")["order_id"].nunique()
    #     exp_norm_fg = (grp_fg / total).fillna(0).rename(f"affinity_{fg}")
    #     exp_norm = pd.concat([exp_norm.rename("exploration_score"),
    #                            exp_norm_fg], axis=1).fillna(0)
        # reset for next iteration — rebuild below cleanly

    # Rebuild cleanly
    total_orders = prior_food.groupby("user_id")["order_id"].nunique().rename("total_orders")
    affinities   = {}
    for fg in ["main", "side", "drink", "snack", "dessert"]:
        cnt = prior_food[prior_food["food_group"] == fg].groupby("user_id")["order_id"].nunique()
        affinities[f"affinity_{fg}"] = (cnt / total_orders).fillna(0)

    # --- reorder variance (measures loyalty vs exploration) ---
    reorder_var = prior_food.groupby("user_id")["reordered"].std().fillna(0).rename("reorder_variance")

    # --- avg basket value and size ---
    basket_stats = prior_food.groupby(["user_id", "order_id"]).agg(
        basket_value=("price", "sum"),
        basket_size=("product_id", "count")
    ).groupby("user_id").mean()

    # --- addon rate: orders with > 1 item / total orders ---
    addon_rate = prior_food.groupby(["user_id", "order_id"])["product_id"].count()
    addon_rate = (addon_rate > 1).groupby(level=0).mean().rename("addon_history_rate")

    # --- combine ---
    exp_series  = exp_scores.pipe(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)).rename("exploration_score")
    users = pd.concat(
        [exp_series, reorder_var, total_orders, addon_rate, basket_stats["basket_value"],
         basket_stats["basket_size"]] + [v for v in affinities.values()],
        axis=1
    ).fillna(0).reset_index()

    print(f"  User profiles built: {len(users):,}")
    print(f"  Avg exploration score: {users['exploration_score'].mean():.3f}")
    return users


def build_instacart_baskets(orders, prior, train_split, items, users):
    """
    Build training pairs using within-basket leave-one-out.
    add_to_cart_order tells us exactly when each item was added.

    For each basket of N items:
      For position i (1 → N-1):
        cart    = items added at positions 1..i-1
        target  = item added at position i  (positive)
        negatives = hard in-basket negatives (items from same dept not in basket)
    """
    print("\n── Building training pairs (within-basket leave-one-out)...")

    food_pids    = set(items["product_id"])
    food_dept    = items.set_index("product_id")["food_group"].to_dict()
    food_price   = items.set_index("product_id")["price"].to_dict()

    # Merge prior + train orders, keep food items only
    all_interactions = pd.concat([
        prior.merge(orders[["order_id", "user_id", "order_hour_of_day",
                             "order_dow", "days_since_prior_order"]], on="order_id"),
        train_split.merge(orders[["order_id", "user_id", "order_hour_of_day",
                                   "order_dow", "days_since_prior_order"]], on="order_id"),
    ])
    all_interactions = all_interactions[all_interactions["product_id"].isin(food_pids)]
    all_interactions = all_interactions.sort_values(["order_id", "add_to_cart_order"])

    # Build item-by-foodgroup index for hard negative sampling
    dept_items = defaultdict(list)
    for pid, fg in food_dept.items():
        dept_items[fg].append(pid)
    dept_items = {k: list(set(v)) for k, v in dept_items.items()}

    # Split users: 80% train, 10% val, 10% test
    user_ids = users["user_id"].values.copy()
    np.random.shuffle(user_ids)
    n         = len(user_ids)
    train_u   = set(user_ids[:int(0.8*n)])
    val_u     = set(user_ids[int(0.8*n):int(0.9*n)])
    test_u    = set(user_ids[int(0.9*n):])

    print(f"  Train users: {len(train_u):,} | Val: {len(val_u):,} | Test: {len(test_u):,}")

    N_NEG = 4   # hard negatives per positive

    def make_pairs(interactions_subset, split_name):
        pairs = []
        orders_grp = interactions_subset.groupby("order_id")

        for order_id, grp in tqdm(orders_grp, desc=f"  {split_name}", leave=False):
            grp = grp.sort_values("add_to_cart_order")
            items_in_basket = grp["product_id"].tolist()
            hour  = grp["order_hour_of_day"].iloc[0]
            dow   = grp["order_dow"].iloc[0]
            uid   = grp["user_id"].iloc[0]

            if len(items_in_basket) < 2:
                continue

            basket_set = set(items_in_basket)

            for i in range(1, len(items_in_basket)):
                cart   = items_in_basket[:i]
                target = items_in_basket[i]

                # Hard negatives: same food group, not in basket
                tgt_fg    = food_dept.get(target)
                if tgt_fg is None:
                    continue
                candidates = [p for p in dept_items[tgt_fg]
                              if p not in basket_set and p != target]
                if len(candidates) == 0:
                    continue
                negatives = np.random.choice(
                    candidates, size=min(N_NEG, len(candidates)), replace=False
                ).tolist()

                # Cart-level aggregate features
                cart_prices   = [food_price.get(p, 4.0) for p in cart]
                cart_fgs      = [food_dept.get(p) for p in cart]
                cart_total    = round(sum(cart_prices), 2)
                cart_size     = len(cart)

                # Meal completion flags from current cart
                cart_fg_set   = set(f for f in cart_fgs if f)
                has_main      = int("main"    in cart_fg_set)
                has_side      = int("side"    in cart_fg_set)
                has_drink     = int("drink"   in cart_fg_set)
                has_snack     = int("snack"   in cart_fg_set)
                has_dessert   = int("dessert" in cart_fg_set)

                # Cross features
                cand_price    = food_price.get(target, 4.0)
                category_gap  = int(tgt_fg not in cart_fg_set)
                price_ratio   = round(cand_price / (cart_total / cart_size + 1e-6), 3)
                price_add_pct = round(cand_price / (cart_total + 1e-6), 3)

                pairs.append({
                    "user_id":        uid,
                    "order_id":       order_id,
                    "cart":           cart,          # list of product_ids
                    "cart_size":      cart_size,
                    "cart_total":     cart_total,
                    "positive":       target,
                    "negatives":      negatives,
                    "food_group":     tgt_fg,
                    "hour":           int(hour),
                    "dow":            int(dow),
                    "meal_period":    hour_to_meal_period(int(hour)),
                    "day_type":       day_to_type(int(dow)),
                    "has_main":       has_main,
                    "has_side":       has_side,
                    "has_drink":      has_drink,
                    "has_snack":      has_snack,
                    "has_dessert":    has_dessert,
                    "category_gap":   category_gap,
                    "price_ratio":    price_ratio,
                    "price_add_pct":  price_add_pct,
                    "cand_price":     round(cand_price, 2),
                })

        return pd.DataFrame(pairs)

    # Partition all_interactions by user split
    train_orders = all_interactions[all_interactions["user_id"].isin(train_u)]
    val_orders   = all_interactions[all_interactions["user_id"].isin(val_u)]
    test_orders  = all_interactions[all_interactions["user_id"].isin(test_u)]

    train_pairs = make_pairs(train_orders, "Train")
    val_pairs   = make_pairs(val_orders,   "Val")
    test_pairs  = make_pairs(test_orders,  "Test")

    print(f"\n  Train pairs: {len(train_pairs):,}")
    print(f"  Val pairs:   {len(val_pairs):,}")
    print(f"  Test pairs:  {len(test_pairs):,}")
    return train_pairs, val_pairs, test_pairs


def compute_pmi_matrix(train_pairs, items):
    """
    Build item-item PMI co-occurrence matrix for the retrieval graph.
    PMI(i, j) = log[ P(i,j) / P(i)P(j) ]
    """
    print("\n── Computing PMI co-occurrence matrix...")

    food_pids  = items["product_id"].tolist()
    pid2idx    = {pid: idx for idx, pid in enumerate(food_pids)}
    N          = len(food_pids)

    item_counts  = defaultdict(int)
    pair_counts  = defaultdict(int)
    total_baskets = 0

    for _, row in tqdm(train_pairs.iterrows(), total=len(train_pairs),
                       desc="  Counting co-occurrences"):
        basket = row["cart"] + [row["positive"]]
        total_baskets += 1
        seen = set()
        for p in basket:
            if p in pid2idx and p not in seen:
                item_counts[p] += 1
                seen.add(p)
        for i, pi in enumerate(basket):
            for pj in basket[i+1:]:
                if pi in pid2idx and pj in pid2idx and pi != pj:
                    key = (min(pi, pj), max(pi, pj))
                    pair_counts[key] += 1

    print(f"  Unique items seen: {len(item_counts):,}")
    print(f"  Unique co-occurring pairs: {len(pair_counts):,}")

    # Build sparse PMI matrix
    rows, cols, data = [], [], []
    for (pi, pj), cnt in pair_counts.items():
        if cnt < 2:
            continue    # skip rare pairs
        pmi = np.log(
            (cnt / total_baskets) /
            ((item_counts[pi] / total_baskets) *
             (item_counts[pj] / total_baskets) + 1e-9)
        )
        if pmi > 0:     # keep positive PMI only
            i, j = pid2idx[pi], pid2idx[pj]
            rows += [i, j]; cols += [j, i]
            data += [pmi, pmi]

    pmi_mat = csr_matrix((data, (rows, cols)), shape=(N, N))
    print(f"  PMI matrix: {N}×{N}, {len(data)//2:,} positive pairs")
    return pmi_mat, pid2idx


def build_text_embeddings(items):
    """
    Run MiniLM over product names to produce semantic item embeddings.
    """
    print("\n── Building MiniLM text embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        names = items["product_name"].fillna("").tolist()
        embs  = model.encode(names, batch_size=256, show_progress_bar=True,
                             convert_to_numpy=True)
        print(f"  Embeddings shape: {embs.shape}")
        return embs
    except ImportError:
        print("  ⚠ sentence-transformers not installed. Run:")
        print("    pip install sentence-transformers")
        print("  Returning zeros as placeholder.")
        return np.zeros((len(items), 384), dtype=np.float32)


# =============================================================================
# DHRD PREPROCESSING
# =============================================================================

def load_dhrd(data_dir):
    """
    DHRD has three city zips: data_se.zip, data_sg.zip, data_tw.zip
    Each contains JSON files with orders: user_id, vendor_id, item_ids, timestamps
    This function reads and normalises them into a common format.
    """
    print("\n" + "="*60)
    print(" LOADING DHRD FILES")
    print("="*60)

    city_files = {
        "SE": os.path.join(data_dir, "dhrd", "data_se.zip"),
        "SG": os.path.join(data_dir, "dhrd", "data_sg.zip"),
        "TW": os.path.join(data_dir, "dhrd", "data_tw.zip"),
    }

    all_orders = []
    for city, fpath in city_files.items():
        if not os.path.exists(fpath):
            print(f"  ⚠ {fpath} not found — skipping {city}")
            continue
        print(f"  Reading {city}...")
        with zipfile.ZipFile(fpath) as zf:
            for fname in zf.namelist():
                with zf.open(fname) as f:
                    try:
                        # DHRD files are typically CSV or JSON — handle both
                        if fname.endswith(".csv"):
                            df = pd.read_csv(f)
                        elif fname.endswith(".json") or fname.endswith(".jsonl"):
                            df = pd.read_json(f, lines=True)
                        else:
                            continue
                        df["city"] = city
                        all_orders.append(df)
                        print(f"    {fname}: {len(df):,} rows")
                    except Exception as e:
                        print(f"    Could not read {fname}: {e}")

    if not all_orders:
        print("  No DHRD files loaded.")
        return pd.DataFrame()

    dhrd = pd.concat(all_orders, ignore_index=True)
    print(f"  Total DHRD rows: {len(dhrd):,}")
    print(f"  Columns: {list(dhrd.columns)}")
    return dhrd


def preprocess_dhrd(dhrd):
    """
    Normalise DHRD into the same pair format as Instacart.
    DHRD column names may vary — adapt if needed after inspecting your files.
    """
    if dhrd.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print("\n── Preprocessing DHRD...")

    # DHRD typical columns: user_id, order_id, item_id, timestamp, vendor_id
    # Rename to standard names if they differ
    rename_map = {}
    for col in dhrd.columns:
        cl = col.lower()
        if "user" in cl:       rename_map[col] = "user_id"
        elif "order" in cl and "id" in cl: rename_map[col] = "order_id"
        elif "item" in cl or "product" in cl: rename_map[col] = "product_id"
        elif "time" in cl or "date" in cl: rename_map[col] = "timestamp"
        elif "vendor" in cl or "rest" in cl: rename_map[col] = "vendor_id"
    dhrd = dhrd.rename(columns=rename_map)

    required = ["user_id", "order_id", "product_id"]
    for col in required:
        if col not in dhrd.columns:
            print(f"  ⚠ Column '{col}' not found. Inspect your DHRD files and adjust rename_map.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Parse timestamps if available
    if "timestamp" in dhrd.columns:
        dhrd["timestamp"] = pd.to_datetime(dhrd["timestamp"], errors="coerce")
        dhrd["hour"] = dhrd["timestamp"].dt.hour.fillna(12).astype(int)
        dhrd["dow"]  = dhrd["timestamp"].dt.dayofweek.fillna(0).astype(int)
    else:
        dhrd["hour"] = 12
        dhrd["dow"]  = 0

    dhrd["meal_period"] = dhrd["hour"].apply(hour_to_meal_period)
    dhrd["day_type"]    = dhrd["dow"].apply(day_to_type)

    # Encode string IDs to integers
    for col in ["user_id", "product_id", "order_id"]:
        if dhrd[col].dtype == object:
            dhrd[col] = dhrd[col].astype("category").cat.codes

    # Build baskets: group by order_id
    baskets = dhrd.groupby("order_id").agg(
        user_id=("user_id", "first"),
        items=("product_id", list),
        hour=("hour", "first"),
        dow=("dow", "first"),
        meal_period=("meal_period", "first"),
        day_type=("day_type", "first"),
    ).reset_index()
    baskets = baskets[baskets["items"].map(len) >= 2]

    # Build pairs (no add_to_cart_order in DHRD — use position as proxy)
    pairs = []
    for _, row in tqdm(baskets.iterrows(), total=len(baskets), desc="  Building DHRD pairs"):
        items_list = row["items"]
        basket_set = set(items_list)
        for i in range(1, len(items_list)):
            cart   = items_list[:i]
            target = items_list[i]
            negs   = [p for p in items_list[i+1:]
                      if p != target and p not in set(cart)][:4]
            if len(negs) == 0:
                continue
            pairs.append({
                "user_id":      row["user_id"],
                "order_id":     row["order_id"],
                "cart":         cart,
                "cart_size":    len(cart),
                "positive":     target,
                "negatives":    negs,
                "hour":         row["hour"],
                "dow":          row["dow"],
                "meal_period":  row["meal_period"],
                "day_type":     row["day_type"],
            })

    pairs = pd.DataFrame(pairs)
    users = pd.Series(pairs["user_id"].unique())
    np.random.shuffle(users.values)
    n = len(users)
    train_u = set(users[:int(0.8*n)])
    val_u   = set(users[int(0.8*n):int(0.9*n)])
    test_u  = set(users[int(0.9*n):])

    train_p = pairs[pairs["user_id"].isin(train_u)]
    val_p   = pairs[pairs["user_id"].isin(val_u)]
    test_p  = pairs[pairs["user_id"].isin(test_u)]
    print(f"  DHRD train: {len(train_p):,} | val: {len(val_p):,} | test: {len(test_p):,}")
    return train_p, val_p, test_p


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(items, users, train_pairs, val_pairs, test_pairs,
                 pmi_mat, text_embs, pid2idx, suffix="instacart"):
    print(f"\n── Saving outputs ({suffix})...")

    # items and users
    items.to_parquet(f"{OUT_DIR}/items_{suffix}.parquet",  index=False)
    users.to_parquet(f"{OUT_DIR}/users_{suffix}.parquet",  index=False)

    # training pairs — store cart as JSON string (parquet doesn't like list-of-lists natively)
    for split, df in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        df = df.copy()
        df["cart"]      = df["cart"].apply(json.dumps)
        df["negatives"] = df["negatives"].apply(json.dumps)
        df.to_parquet(f"{OUT_DIR}/{split}_pairs_{suffix}.parquet", index=False)
        print(f"  Saved {split}_pairs_{suffix}.parquet: {len(df):,} rows")

    # PMI matrix
    if pmi_mat is not None:
        save_npz(f"{OUT_DIR}/pmi_matrix_{suffix}.npz", pmi_mat)
        print(f"  Saved pmi_matrix_{suffix}.npz")

    # Text embeddings
    if text_embs is not None:
        np.save(f"{OUT_DIR}/text_embeddings_{suffix}.npy", text_embs)
        print(f"  Saved text_embeddings_{suffix}.npy: {text_embs.shape}")

    # ID maps
    with open(f"{OUT_DIR}/pid2idx_{suffix}.json", "w") as f:
        json.dump({str(k): v for k, v in pid2idx.items()}, f)

    # Dataset stats
    stats = {
        "dataset":       suffix,
        "n_items":       len(items),
        "n_users":       len(users),
        "train_pairs":   len(train_pairs),
        "val_pairs":     len(val_pairs),
        "test_pairs":    len(test_pairs),
        "food_groups":   items["food_group"].value_counts().to_dict() if "food_group" in items else {},
        "positive_rate": round(1 / (1 + 4), 4),   # 1 pos : 4 neg
    }
    with open(f"{OUT_DIR}/stats_{suffix}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats_{suffix}.json")
    print(f"\n  ✓ All outputs written to ./{OUT_DIR}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline")
    parser.add_argument("--dataset",  choices=["instacart", "dhrd", "both"], default="instacart")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory containing instacart/ and dhrd/ subfolders")
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="Skip MiniLM text embedding step (saves ~5 min on CPU)")
    args = parser.parse_args()

    if args.dataset in ["instacart", "both"]:
        inst_dir = os.path.join(args.data_dir, "instacart")

        aisles, departments, products, orders, prior, train_split = load_instacart(inst_dir)
        items   = build_instacart_items(products, aisles, departments)
        users   = build_instacart_users(orders, prior, items)
        train_p, val_p, test_p = build_instacart_baskets(orders, prior, train_split, items, users)
        pmi_mat, pid2idx       = compute_pmi_matrix(train_p, items)
        text_embs              = None if args.skip_embeddings else build_text_embeddings(items)

        save_outputs(items, users, train_p, val_p, test_p,
                     pmi_mat, text_embs, pid2idx, suffix="instacart")

    if args.dataset in ["dhrd", "both"]:
        dhrd     = load_dhrd(args.data_dir)
        tr, va, te = preprocess_dhrd(dhrd)

        if not tr.empty:
            # For DHRD we don't have department → food group mapping,
            # so items table is a simple product ID list
            dhrd_items = pd.DataFrame({
                "product_id": sorted(set(tr["positive"].tolist() +
                                         sum(tr["cart"].tolist(), [])))
            })
            dhrd_users = pd.DataFrame({"user_id": tr["user_id"].unique()})
            pmi_dhrd, pid2idx_dhrd = compute_pmi_matrix(tr, dhrd_items.rename(columns={"product_id": "product_id"}))
            save_outputs(dhrd_items, dhrd_users, tr, va, te,
                         pmi_dhrd, None, pid2idx_dhrd, suffix="dhrd")


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE GUIDE
# =============================================================================
"""
FOLDER STRUCTURE EXPECTED:
─────────────────────────────────────────────────────────
data/
├── instacart/
│   ├── aisles.csv
│   ├── departments.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── orders.csv
│   └── products.csv
└── dhrd/
    ├── data_se.zip
    ├── data_sg.zip
    └── data_tw.zip

COMMANDS:
─────────────────────────────────────────────────────────
# Install dependencies
pip install pandas numpy scipy sentence-transformers tqdm pyarrow

# Run Instacart only (fastest, ~10-15 min)
python3.11 Data_processing.py --dataset instacart --data_dir ./data

# Run both datasets
python3.11 Data_processing.py --dataset both --data_dir ./data

# Skip text embeddings (saves time, add later)
python3.11 Data_processing.py --dataset instacart --data_dir ./data --skip_embeddings

OUTPUTS:
─────────────────────────────────────────────────────────
outputs/
├── items_instacart.parquet         ← ~38k items with price, food_group, dept
├── users_instacart.parquet         ← 206k users with exploration_score, affinities
├── train_pairs_instacart.parquet   ← ~8M (cart, positive, negatives) pairs
├── val_pairs_instacart.parquet     ← ~1M pairs
├── test_pairs_instacart.parquet    ← ~1M pairs
├── pmi_matrix_instacart.npz        ← sparse item co-occurrence matrix
├── text_embeddings_instacart.npy   ← (38k, 384) MiniLM embeddings
├── pid2idx_instacart.json          ← product_id → integer index
└── stats_instacart.json            ← dataset statistics

LOADING IN YOUR MODEL:
─────────────────────────────────────────────────────────
import pandas as pd, numpy as np, json
from scipy.sparse import load_npz

items   = pd.read_parquet("outputs/items_instacart.parquet")
users   = pd.read_parquet("outputs/users_instacart.parquet")
train   = pd.read_parquet("outputs/train_pairs_instacart.parquet")

# Decode cart and negatives back from JSON strings
train["cart"]      = train["cart"].apply(json.loads)
train["negatives"] = train["negatives"].apply(json.loads)

pmi_matrix  = load_npz("outputs/pmi_matrix_instacart.npz")
text_embs   = np.load("outputs/text_embeddings_instacart.npy")
pid2idx     = json.load(open("outputs/pid2idx_instacart.json"))
"""