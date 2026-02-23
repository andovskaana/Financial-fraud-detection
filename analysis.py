import pandas as pd

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv(
    "financial_fraud_detection_dataset.csv",
    parse_dates=["timestamp"]  # parse datetime immediately
)

print("Dataset shape:", df.shape)

# ---------------------------
# 2. Missing values by column
# ---------------------------
missing_summary = (
    df.isna()
    .sum()
    .to_frame(name="missing_count")
    .assign(missing_percent=lambda x: (x["missing_count"] / len(df)) * 100)
    .sort_values("missing_count", ascending=False)
)

print("\nMissing values by column:")
print(missing_summary)

# ---------------------------
# 3. Average transaction amount by user
# (using sender_account as the user)
# ---------------------------
avg_amount_by_user = (
    df.groupby("sender_account")["amount"]
    .mean()
    .reset_index(name="avg_transaction_amount")
)

print("\nAverage transaction amount by user:")
print(avg_amount_by_user.head())

# ---------------------------
# 4. Transaction frequency by user
# ---------------------------
transaction_freq_by_user = (
    df.groupby("sender_account")["transaction_id"]
    .count()
    .reset_index(name="transaction_count")
)

print("\nTransaction frequency by user:")
print(transaction_freq_by_user.head())

# ---------------------------
# 5. Datetime feature engineering
# ---------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df["hour"] = df["timestamp"].dt.hour.astype("Int64")
df["day"] = df["timestamp"].dt.day.astype("Int64")
df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("Int64")


df["week"] = df["timestamp"].dt.isocalendar().week.astype("Int64")

df["month"] = df["timestamp"].dt.month.astype("Int64")
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int64")

print("\nDatetime features added:")
print(df[["timestamp", "hour", "day", "day_of_week", "month", "is_weekend"]].head())

# ---------------------------
# 6. (Optional) Merge user features back into main dataframe
# ---------------------------
df = df.merge(avg_amount_by_user, on="sender_account", how="left")
df = df.merge(transaction_freq_by_user, on="sender_account", how="left")

print("\nFinal dataset shape with engineered features:", df.shape)
df["fraud"] = df["is_fraud"].fillna(False).astype("int64")
print(df["fraud"].value_counts())



# ---------------------------
# 7. Save EDA-enhanced dataset (optional)
# ---------------------------
#df.to_csv("fraud_dataset_eda_features.csv", index=False)
