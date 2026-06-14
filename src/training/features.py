"""
Feature Engineering Module for Fraud Detection

This module provides both batch and streaming-compatible feature engineering.
It automatically adapts to available columns in the dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict
import json
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureConfig:
    """Configuration for feature engineering - tracks available columns and settings."""

    # Core columns (required)
    transaction_id_col: str = "transaction_id"
    timestamp_col: str = "timestamp"
    sender_col: str = "sender_account"
    amount_col: str = "amount"
    target_col: str = "is_fraud"

    # Optional columns (auto-detected)
    receiver_col: Optional[str] = None
    transaction_type_col: Optional[str] = None
    sender_country_col: Optional[str] = None
    receiver_country_col: Optional[str] = None

    # Detected categorical columns
    categorical_cols: List[str] = field(default_factory=list)

    # Feature settings
    velocity_windows_minutes: List[int] = field(default_factory=lambda: [5, 60, 1440])  # 5m, 1h, 24h
    max_history_per_user: int = 20  # Bounded state for streaming

    # Encoding mappings (filled during training)
    label_encoders: Dict[str, Dict] = field(default_factory=dict)

    # Feature statistics for normalization
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Final feature columns used in model
    feature_columns: List[str] = field(default_factory=list)

    # Model threshold for anomaly detection
    anomaly_threshold: float = 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'transaction_id_col': self.transaction_id_col,
            'timestamp_col': self.timestamp_col,
            'sender_col': self.sender_col,
            'amount_col': self.amount_col,
            'target_col': self.target_col,
            'receiver_col': self.receiver_col,
            'transaction_type_col': self.transaction_type_col,
            'sender_country_col': self.sender_country_col,
            'receiver_country_col': self.receiver_country_col,
            'categorical_cols': self.categorical_cols,
            'velocity_windows_minutes': self.velocity_windows_minutes,
            'max_history_per_user': self.max_history_per_user,
            'label_encoders': self.label_encoders,
            'feature_stats': self.feature_stats,
            'feature_columns': self.feature_columns,
            'anomaly_threshold': self.anomaly_threshold
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureConfig':
        """Load from dictionary."""
        return cls(**data)

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FeatureConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class ColumnDetector:
    """Auto-detect available columns in the dataset."""

    COLUMN_PATTERNS = {
        'transaction_id': ['transaction_id', 'trans_id', 'txn_id', 'id'],
        'timestamp': ['timestamp', 'trans_date', 'transaction_date', 'date', 'datetime', 'time'],
        'sender_account': ['sender_account', 'sender', 'from_account', 'source_account', 'account_id', 'user_id'],
        'receiver_account': ['receiver_account', 'receiver', 'to_account', 'dest_account', 'destination_account'],
        'amount': ['amount', 'transaction_amount', 'value', 'amt'],
        'transaction_type': ['transaction_type', 'type', 'trans_type', 'category'],
        'sender_country': ['sender_country', 'from_country', 'source_country', 'country', 'location'],
        'receiver_country': ['receiver_country', 'to_country', 'dest_country', 'destination_country'],
        'is_fraud': ['is_fraud', 'fraud', 'label', 'target', 'is_fraudulent', 'fraudulent']
    }

    @classmethod
    def detect_columns(cls, df: pd.DataFrame) -> FeatureConfig:
        """Detect and map columns from the dataset."""
        config = FeatureConfig()
        available_cols = [c.lower() for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}

        for field_name, patterns in cls.COLUMN_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in available_cols:
                    original_col = col_map[pattern.lower()]
                    if field_name == 'transaction_id':
                        config.transaction_id_col = original_col
                    elif field_name == 'timestamp':
                        config.timestamp_col = original_col
                    elif field_name == 'sender_account':
                        config.sender_col = original_col
                    elif field_name == 'receiver_account':
                        config.receiver_col = original_col
                    elif field_name == 'amount':
                        config.amount_col = original_col
                    elif field_name == 'transaction_type':
                        config.transaction_type_col = original_col
                    elif field_name == 'sender_country':
                        config.sender_country_col = original_col
                    elif field_name == 'receiver_country':
                        config.receiver_country_col = original_col
                    elif field_name == 'is_fraud':
                        config.target_col = original_col
                    break

        # Detect additional categorical columns
        categorical_candidates = []
        for col in df.columns:
            if col.lower() not in [p.lower() for patterns in cls.COLUMN_PATTERNS.values() for p in patterns]:
                if df[col].dtype == 'object' or df[col].nunique() < 50:
                    if col not in [config.transaction_id_col, config.timestamp_col,
                                   config.sender_col, config.amount_col]:
                        categorical_candidates.append(col)

        config.categorical_cols = categorical_candidates
        return config


class BatchFeatureEngineer:
    """Feature engineering for batch (training) mode."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create datetime-based features."""
        df = df.copy()
        ts_col = self.config.timestamp_col

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

        # Extract features
        df['hour'] = df[ts_col].dt.hour.fillna(0).astype(int)
        df['day'] = df[ts_col].dt.day.fillna(1).astype(int)
        df['day_of_week'] = df[ts_col].dt.dayofweek.fillna(0).astype(int)
        df['week'] = df[ts_col].dt.isocalendar().week.fillna(1).astype(int)
        df['month'] = df[ts_col].dt.month.fillna(1).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Time-based risk features
        df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

        return df

    def create_user_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-level aggregate features."""
        df = df.copy()
        sender_col = self.config.sender_col
        amount_col = self.config.amount_col

        # Basic aggregates per user
        user_stats = df.groupby(sender_col)[amount_col].agg([
            ('user_avg_amount', 'mean'),
            ('user_std_amount', 'std'),
            ('user_min_amount', 'min'),
            ('user_max_amount', 'max'),
            ('user_transaction_count', 'count')
        ]).reset_index()

        # Fill NaN std with 0
        user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(0)

        df = df.merge(user_stats, on=sender_col, how='left')

        # Previous fraud-labelled transactions for this user.
        # This is computed chronologically and shifted so the current row's
        # label is not counted as part of its own history. If the target is
        # not available, e.g. inference-only transform data, default to 0.
        target_col = self.config.target_col
        ts_col = self.config.timestamp_col
        if target_col in df.columns:
            fraud_labels = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
            fraud_history_frame = df[[sender_col, ts_col]].copy()
            fraud_history_frame['_original_index'] = df.index
            fraud_history_frame['_fraud_label'] = fraud_labels.values
            fraud_history_frame = fraud_history_frame.sort_values([sender_col, ts_col, '_original_index'])
            fraud_history_frame['prev_fraud_count'] = (
                fraud_history_frame
                .groupby(sender_col)['_fraud_label']
                .cumsum()
                .shift(fill_value=0)
            )

            # The global shift above can carry the previous user's last count
            # into the first row of the next user, so force each first user row to 0.
            first_for_user = fraud_history_frame[sender_col].ne(fraud_history_frame[sender_col].shift())
            fraud_history_frame.loc[first_for_user, 'prev_fraud_count'] = 0

            df['prev_fraud_count'] = (
                fraud_history_frame
                .set_index('_original_index')['prev_fraud_count']
                .reindex(df.index)
                .fillna(0)
                .astype(int)
            )
        else:
            df['prev_fraud_count'] = 0

        # Amount relative to user's average
        df['amount_to_user_avg_ratio'] = df[amount_col] / (df['user_avg_amount'] + 1e-6)
        df['amount_zscore'] = (df[amount_col] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-6)

        return df

    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ OPTIMIZED: Create velocity/rolling window features using pre-allocated arrays.
        Bypasses slow pandas row-by-row lookups for blazing fast execution.
        """
        df = df.copy()
        sender_col = self.config.sender_col
        amount_col = self.config.amount_col
        ts_col = self.config.timestamp_col

        # Sort by user and timestamp
        df = df.sort_values([sender_col, ts_col]).reset_index(drop=True)

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

        n_rows = len(df)

        # Pre-allocate dictionaries of numpy arrays to completely replace slow df.loc
        velocity_data = {}
        for window_min in self.config.velocity_windows_minutes:
            velocity_data[f'tx_count_{window_min}m'] = np.zeros(n_rows, dtype=np.int32)
            velocity_data[f'amount_sum_{window_min}m'] = np.zeros(n_rows, dtype=np.float32)
            velocity_data[f'amount_avg_{window_min}m'] = np.zeros(n_rows, dtype=np.float32)

        time_since_last_tx = np.zeros(n_rows, dtype=np.float32)
        amount_to_rolling_avg_ratio = np.ones(n_rows, dtype=np.float32)

        has_receiver = self.config.receiver_col and self.config.receiver_col in df.columns
        unique_receivers_1h = np.zeros(n_rows, dtype=np.int32) if has_receiver else None

        # Extract underlying values to numpy arrays for raw execution speed
        timestamps = df[ts_col].values.astype('datetime64[ns]').astype(np.int64)  # unix nanoseconds
        amounts = df[amount_col].values.astype(np.float32)
        receivers = df[self.config.receiver_col].values if has_receiver else None

        max_history = self.config.max_history_per_user

        # Iterate over groups, but perform calculations using positional indexing
        for _, group in df.groupby(sender_col):
            indices = group.index.values
            if len(indices) == 0:
                continue

            group_ts = timestamps[indices]
            group_amounts = amounts[indices]
            group_receivers = receivers[indices] if has_receiver else None

            # Calculate time difference between consecutive transactions instantly via vectorization
            if len(indices) > 1:
                time_since_last_tx[indices[1:]] = (group_ts[1:] - group_ts[:-1]) / 1e9

            # Loop through history using the bounded constraint window
            for i, idx in enumerate(indices):
                current_ts = group_ts[i]
                lookback_start = max(0, i - max_history)

                # Slice arrays directly using raw slices
                sub_ts = group_ts[lookback_start:i]
                sub_amounts = group_amounts[lookback_start:i]

                for window_min in self.config.velocity_windows_minutes:
                    window_ns = int(window_min * 60 * 1e9)

                    # Compute window mask over slice element via fast array evaluation
                    mask = (current_ts - sub_ts) <= window_ns
                    window_amounts = sub_amounts[mask]
                    n_tx = len(window_amounts)

                    if n_tx > 0:
                        amt_sum = window_amounts.sum()
                        velocity_data[f'tx_count_{window_min}m'][idx] = n_tx
                        velocity_data[f'amount_sum_{window_min}m'][idx] = amt_sum
                        velocity_data[f'amount_avg_{window_min}m'][idx] = amt_sum / n_tx

                        # Unique receivers inside 1h window
                        if window_min == 60 and has_receiver:
                            window_receivers = group_receivers[lookback_start:i][mask]
                            unique_receivers_1h[idx] = len(set(window_receivers))

                # Amount to rolling average ratio (24h)
                rolling_avg = velocity_data['amount_avg_1440m'][idx]
                if rolling_avg > 0:
                    amount_to_rolling_avg_ratio[idx] = group_amounts[i] / rolling_avg

        # Assign arrays all at once back to DataFrame
        df['time_since_last_tx'] = time_since_last_tx
        df['amount_to_rolling_avg_ratio'] = amount_to_rolling_avg_ratio
        if has_receiver:
            df['unique_receivers_1h'] = unique_receivers_1h

        for col_name, array_values in velocity_data.items():
            df[col_name] = array_values

        return df

    def create_country_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create country-based features if available."""
        df = df.copy()

        if self.config.sender_country_col and self.config.sender_country_col in df.columns:
            if self.config.receiver_country_col and self.config.receiver_country_col in df.columns:
                # Cross-border transaction flag
                df['is_cross_border'] = (
                    df[self.config.sender_country_col] != df[self.config.receiver_country_col]
                ).astype(int)

        return df

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical columns with label encoding."""
        df = df.copy()

        # Transaction type
        if self.config.transaction_type_col and self.config.transaction_type_col in df.columns:
            col = self.config.transaction_type_col
            df[col] = df[col].fillna('unknown').astype(str)

            if fit:
                unique_vals = df[col].unique().tolist()
                self.config.label_encoders[col] = {v: i for i, v in enumerate(unique_vals)}

            mapping = self.config.label_encoders.get(col, {})
            df[f'{col}_encoded'] = df[col].map(mapping).fillna(-1).astype(int)

        # Countries
        for country_col in [self.config.sender_country_col, self.config.receiver_country_col]:
            if country_col and country_col in df.columns:
                df[country_col] = df[country_col].fillna('unknown').astype(str)

                if fit:
                    unique_vals = df[country_col].unique().tolist()
                    self.config.label_encoders[country_col] = {v: i for i, v in enumerate(unique_vals)}

                mapping = self.config.label_encoders.get(country_col, {})
                df[f'{country_col}_encoded'] = df[country_col].map(mapping).fillna(-1).astype(int)

        # Additional categorical columns
        for col in self.config.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)

                if fit:
                    unique_vals = df[col].unique().tolist()
                    self.config.label_encoders[col] = {v: i for i, v in enumerate(unique_vals)}

                mapping = self.config.label_encoders.get(col, {})
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(-1).astype(int)

        return df

    def create_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute batch equivalents of the two streaming post-prediction rules so
        they can be used as model *input features* during training.

        geo_velocity_flag (int 0/1)
            Mirrors the streaming geographic rule: 1 if the sender's country
            changed compared to their immediately previous transaction AND that
            previous transaction was within geo_window_seconds (1 h).

        seq_fraud_flag (int 0/1)
            Mirrors the streaming sequential rule: 1 if the user had at least 2
            consecutive fraud-labelled transactions immediately before the
            current one.  Requires the target column to be present (training
            only); defaults to 0 when labels are unavailable.

        Both features are computed in strict chronological order per user so no
        future information leaks into the feature value of any given row.
        """
        df = df.copy()
        sender_col = self.config.sender_col
        ts_col = self.config.timestamp_col
        target_col = self.config.target_col
        geo_window_seconds = 3600  # mirror streaming geo_window_seconds

        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

        df = df.sort_values([sender_col, ts_col]).reset_index(drop=True)

        # --- geo_velocity_flag ---
        geo_flag = np.zeros(len(df), dtype=np.int32)
        if self.config.sender_country_col and self.config.sender_country_col in df.columns:
            ts_ns = df[ts_col].values.astype('datetime64[ns]').astype(np.int64)
            country_vals = df[self.config.sender_country_col].values
            sender_vals = df[sender_col].values
            prev_ts: dict = {}
            prev_country: dict = {}
            for i in range(len(df)):
                uid = sender_vals[i]
                if uid in prev_ts:
                    elapsed_s = (ts_ns[i] - prev_ts[uid]) / 1e9
                    if elapsed_s <= geo_window_seconds and prev_country[uid] != country_vals[i]:
                        geo_flag[i] = 1
                prev_ts[uid] = ts_ns[i]
                prev_country[uid] = country_vals[i]

        df['geo_velocity_flag'] = geo_flag

        # --- seq_fraud_flag ---
        # For each transaction: 1 if the user had >= 2 consecutive fraud labels
        # immediately before this transaction (resets to 0 on any non-fraud tx).
        seq_flag = np.zeros(len(df), dtype=np.int32)
        if target_col in df.columns:
            fraud_vals = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int).values
            sender_vals = df[sender_col].values
            consecutive: dict = {}
            for i in range(len(df)):
                uid = sender_vals[i]
                count = consecutive.get(uid, 0)
                seq_flag[i] = 1 if count >= 2 else 0
                consecutive[uid] = count + 1 if fraud_vals[i] == 1 else 0

        df['seq_fraud_flag'] = seq_flag

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for model training."""
        feature_cols = []

        # Amount
        feature_cols.append(self.config.amount_col)

        # Datetime features
        datetime_features = ['hour', 'day', 'day_of_week', 'week', 'month', 'is_weekend', 'is_night']
        feature_cols.extend([c for c in datetime_features if c in df.columns])

        # User aggregates
        user_features = ['user_avg_amount', 'user_std_amount', 'user_min_amount',
                        'user_max_amount', 'user_transaction_count', 'prev_fraud_count',
                        'amount_to_user_avg_ratio', 'amount_zscore']
        feature_cols.extend([c for c in user_features if c in df.columns])

        # Velocity features
        for window_min in self.config.velocity_windows_minutes:
            velocity_features = [
                f'tx_count_{window_min}m',
                f'amount_sum_{window_min}m',
                f'amount_avg_{window_min}m'
            ]
            feature_cols.extend([c for c in velocity_features if c in df.columns])

        feature_cols.extend([c for c in ['time_since_last_tx', 'amount_to_rolling_avg_ratio',
                                         'unique_receivers_1h'] if c in df.columns])

        # Country features
        if 'is_cross_border' in df.columns:
            feature_cols.append('is_cross_border')

        # Rule-derived features (only present when create_rule_features() was called)
        for rule_col in ['geo_velocity_flag', 'seq_fraud_flag']:
            if rule_col in df.columns:
                feature_cols.append(rule_col)

        # Encoded categoricals
        encoded_cols = [c for c in df.columns if c.endswith('_encoded')]
        feature_cols.extend(encoded_cols)

        return feature_cols

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Apply all feature engineering steps for training."""
        print("Creating datetime features...")
        df = self.create_datetime_features(df)

        print("Creating user aggregates...")
        df = self.create_user_aggregates(df)

        print("Creating velocity features (Optimized)...")
        df = self.create_velocity_features(df)

        print("Creating country features...")
        df = self.create_country_features(df)

        print("Encoding categorical features...")
        df = self.encode_categoricals(df, fit=True)

        # Get and save feature columns
        feature_cols = self.get_feature_columns(df)
        self.config.feature_columns = feature_cols

        # Handle missing values
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        print(f"Total features: {len(feature_cols)}")
        return df, feature_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering for inference (no fitting)."""
        df = self.create_datetime_features(df)
        df = self.create_user_aggregates(df)
        df = self.create_velocity_features(df)
        df = self.create_country_features(df)
        df = self.encode_categoricals(df, fit=False)

        # Handle missing values
        for col in self.config.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df


class StreamingFeatureState:
    """
    Bounded state for streaming feature computation.
    Maintains per-user history with TTL and max events.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.user_states: Dict[str, Dict] = defaultdict(lambda: {
            'transactions': [],  # List of (timestamp, amount, receiver)
            'total_amount': 0.0,
            'total_count': 0,
            # Number of previous fraud-labelled/anomaly transactions for this user.
            # Updated after prediction, so the current transaction never counts itself.
            'fraud_count': 0
        })
        self.max_history = config.max_history_per_user
        self.ttl_seconds = 24 * 60 * 60  # 24 hours TTL

    def _cleanup_old_transactions(self, user_id: str, current_ts: pd.Timestamp):
        """Remove transactions older than TTL."""
        state = self.user_states[user_id]
        cutoff = current_ts - pd.Timedelta(seconds=self.ttl_seconds)

        # Filter out old transactions
        state['transactions'] = [
            t for t in state['transactions']
            if t[0] >= cutoff
        ][-self.max_history:]  # Keep only last N

    def compute_features(self, transaction: Dict) -> Dict[str, Any]:
        """Compute streaming features for a single transaction."""
        user_id = transaction[self.config.sender_col]
        amount = float(transaction[self.config.amount_col])

        # Parse timestamp
        ts = transaction[self.config.timestamp_col]
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)

        # Cleanup old transactions
        self._cleanup_old_transactions(user_id, ts)

        state = self.user_states[user_id]
        transactions = state['transactions']

        features = {}

        # Datetime features
        features['hour'] = ts.hour
        features['day'] = ts.day
        features['day_of_week'] = ts.dayofweek
        features['week'] = ts.isocalendar()[1]
        features['month'] = ts.month
        features['is_weekend'] = 1 if ts.dayofweek in [5, 6] else 0
        features['is_night'] = 1 if ts.hour in [0, 1, 2, 3, 4, 5, 22, 23] else 0

        # Amount
        features[self.config.amount_col] = amount

        # Previous fraud-labelled transactions for this user.
        # This is read before the current prediction updates fraud_count.
        features['prev_fraud_count'] = int(state.get('fraud_count', 0))

        # Time since last transaction
        if transactions:
            last_ts = transactions[-1][0]
            features['time_since_last_tx'] = (ts - last_ts).total_seconds()
        else:
            features['time_since_last_tx'] = 0.0

        # Velocity features from bounded history
        for window_min in self.config.velocity_windows_minutes:
            window_start = ts - pd.Timedelta(minutes=window_min)

            window_txs = [t for t in transactions if t[0] >= window_start]

            features[f'tx_count_{window_min}m'] = len(window_txs)
            features[f'amount_sum_{window_min}m'] = sum(t[1] for t in window_txs)
            features[f'amount_avg_{window_min}m'] = (
                features[f'amount_sum_{window_min}m'] / len(window_txs)
                if window_txs else 0.0
            )

        # Unique receivers in last hour
        if self.config.receiver_col:
            hour_start = ts - pd.Timedelta(hours=1)
            hour_txs = [t for t in transactions if t[0] >= hour_start]
            unique_receivers = len(set(t[2] for t in hour_txs if t[2] is not None))
            features['unique_receivers_1h'] = unique_receivers

        # Amount to rolling average ratio
        rolling_avg = features.get('amount_avg_1440m', 0)
        features['amount_to_rolling_avg_ratio'] = amount / rolling_avg if rolling_avg > 0 else 1.0

        # User-level stats (from current state)
        all_amounts = [t[1] for t in transactions]
        if all_amounts:
            features['user_avg_amount'] = np.mean(all_amounts)
            features['user_std_amount'] = np.std(all_amounts) if len(all_amounts) > 1 else 0.0
            features['user_min_amount'] = min(all_amounts)
            features['user_max_amount'] = max(all_amounts)
            features['user_transaction_count'] = len(all_amounts)
            features['amount_to_user_avg_ratio'] = amount / features['user_avg_amount']
            features['amount_zscore'] = (
                (amount - features['user_avg_amount']) / (features['user_std_amount'] + 1e-6)
            )
        else:
            features['user_avg_amount'] = amount
            features['user_std_amount'] = 0.0
            features['user_min_amount'] = amount
            features['user_max_amount'] = amount
            features['user_transaction_count'] = 0
            features['amount_to_user_avg_ratio'] = 1.0
            features['amount_zscore'] = 0.0

        # Country features
        if self.config.sender_country_col and self.config.receiver_country_col:
            sender_country = transaction.get(self.config.sender_country_col)
            receiver_country = transaction.get(self.config.receiver_country_col)
            features['is_cross_border'] = 1 if sender_country != receiver_country else 0

        # Encode categoricals
        if self.config.transaction_type_col:
            tx_type = transaction.get(self.config.transaction_type_col, 'unknown')
            mapping = self.config.label_encoders.get(self.config.transaction_type_col, {})
            features[f'{self.config.transaction_type_col}_encoded'] = mapping.get(tx_type, -1)

        for country_col in [self.config.sender_country_col, self.config.receiver_country_col]:
            if country_col:
                country = transaction.get(country_col, 'unknown')
                mapping = self.config.label_encoders.get(country_col, {})
                features[f'{country_col}_encoded'] = mapping.get(country, -1)

        # Update state with current transaction
        receiver = transaction.get(self.config.receiver_col) if self.config.receiver_col else None
        state['transactions'].append((ts, amount, receiver))
        state['total_amount'] += amount
        state['total_count'] += 1

        return features

    def update_fraud_count(self, transaction: Dict, is_fraud: bool):
        """Update per-user previous fraud count after the prediction/label is known.

        The value exposed as prev_fraud_count must represent only transactions
        before the current one, so this method is intentionally called after
        model/ evaluation. In streaming, the prediction is the available
        fraud label; in offline replay, pass the known ground-truth label.
        """
        if not is_fraud:
            return

        user_id = transaction.get(self.config.sender_col)
        if user_id is None:
            return

        state = self.user_states[user_id]
        state['fraud_count'] = int(state.get('fraud_count', 0)) + 1

    def get_feature_vector(self, transaction: Dict) -> np.ndarray:
        """Get feature vector in the same order as training."""
        features = self.compute_features(transaction)

        vector = []
        for col in self.config.feature_columns:
            vector.append(features.get(col, 0.0))

        return np.array(vector, dtype=np.float32)