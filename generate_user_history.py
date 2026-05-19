import os
import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = "outputs"
DATA_DIR = "./data/instacart"

print("=" * 60)
print(" GENERATING USER HISTORY SEQUENCES")
print("=" * 60)

# ---------------------------------------------------
# Load required raw files only
# ---------------------------------------------------

orders = pd.read_csv(
    os.path.join(DATA_DIR, "orders.csv")
)

prior = pd.read_csv(
    os.path.join(DATA_DIR, "order_products__prior.csv")
)

# ---------------------------------------------------
# Optional: match downsampled users
# ---------------------------------------------------

users = pd.read_parquet(
    os.path.join(OUTPUT_DIR, "users_instacart.parquet")
)

valid_users = set(users["user_id"])

orders = orders[
    orders["user_id"].isin(valid_users)
]

valid_order_ids = set(orders["order_id"])

prior = prior[
    prior["order_id"].isin(valid_order_ids)
]

# ---------------------------------------------------
# Merge order metadata
# ---------------------------------------------------

interactions = prior.merge(
    orders[
        ["order_id", "user_id"]
    ],
    on="order_id"
)

# ---------------------------------------------------
# Preserve sequence order
# ---------------------------------------------------

interactions = interactions.sort_values(
    ["user_id", "order_id", "add_to_cart_order"]
)

# ---------------------------------------------------
# Build basket-level sequences
# ---------------------------------------------------

print("\n── Building basket sequences...")

basket_sequences = (
    interactions
    .groupby(["user_id", "order_id"])["product_id"]
    .apply(list)
    .reset_index()
)

# ---------------------------------------------------
# Build user histories
# ---------------------------------------------------

print("── Building user histories...")

user_histories = (
    basket_sequences
    .groupby("user_id")["product_id"]
    .apply(list)
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

save_path = os.path.join(
    OUTPUT_DIR,
    "user_histories_instacart.pkl"
)

user_histories.to_pickle(save_path)

print(f"\n✓ Saved: {save_path}")
print(f"✓ Users: {len(user_histories):,}")

print("\nDONE.")