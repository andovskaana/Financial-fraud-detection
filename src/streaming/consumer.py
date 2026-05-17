"""
Fraud Detection Consumer

Consumes transactions from Kafka, performs inference, and routes
to normal/anomaly topics.

Two modes:
1. In-process: Load model locally for maximum throughput
2. API mode: Call FastAPI predict service (for scaling)
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import joblib
import requests
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.streaming.kafka_io import (
    TransactionConsumer, TransactionRouter, KafkaConfig
)
from src.training.features import FeatureConfig, StreamingFeatureState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetector:
    """
    In-process fraud detector for maximum throughput.
    Loads model locally and maintains streaming feature state.
    """

    def __init__(
        self,
        model_path: str = 'models/fraud_model.joblib',
        config_path: str = 'models/feature_config.json'
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.feature_state = None
        self.model_version = "1.0"
        self._lock = threading.Lock()
        # State for sequential and geographic rules
        self.user_consecutive_anomalies = defaultdict(int)
        self.user_last_tx_info: Dict[str, tuple] = {}
        self.sequential_threshold = 2      # flag anomaly after 2 previous anomalies
        self.geo_window_seconds = 3600     # one-hour window for country changes
        # R13: per-prediction SHAP support for the local consumer path.
        # This is needed because docker-compose runs fraud-consumer in local mode,
        # so the FastAPI predict-service SHAP endpoint is not used by default.
        self._xgb_native_shap = False
        self.explainer = None

    def load(self):
        """Load model and initialize feature state."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info(f"Loading config from {self.config_path}")
        self.config = FeatureConfig.load(self.config_path)

        self.feature_state = StreamingFeatureState(self.config)

        # R13: enable SHAP explanations in the local streaming consumer.
        # Prefer XGBoost native pred_contribs=True because it is exact Tree SHAP
        # and avoids SHAP/XGBoost base_score parsing issues in some versions.
        try:
            import xgboost as xgb
            if isinstance(self.model, xgb.XGBClassifier):
                self._xgb_native_shap = True
                logger.info("Local SHAP enabled via XGBoost native pred_contribs=True")
            else:
                import shap
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Local SHAP enabled via shap.TreeExplainer")
        except Exception as shap_err:
            logger.warning(f"Local SHAP disabled: {shap_err}")
            self._xgb_native_shap = False
            self.explainer = None

        logger.info(f"Detector ready. Threshold: {self.config.anomaly_threshold}")
        return self

    def _compute_shap_values(self, feature_vector: np.ndarray) -> Optional[np.ndarray]:
        """Compute one SHAP contribution per configured feature.

        Returns None if SHAP is unavailable. For XGBoost, native
        pred_contribs=True returns exact Tree SHAP values plus a final bias
        column; the bias column is removed before returning.
        """
        try:
            x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
            feature_cols = self.config.feature_columns

            if self._xgb_native_shap:
                import xgboost as xgb
                dmat = xgb.DMatrix(x, feature_names=feature_cols)
                contribs = self.model.get_booster().predict(
                    dmat, pred_contribs=True, validate_features=False
                )
                if contribs.ndim == 3:
                    if contribs.shape[1] == len(feature_cols) + 1:
                        contribs = contribs[:, :, 1]
                    else:
                        contribs = contribs[:, 1, :]
                return np.asarray(contribs[0, :-1], dtype=np.float64)

            if self.explainer is not None:
                sv = self.explainer.shap_values(x)
                if isinstance(sv, list):
                    sv = sv[1]
                if hasattr(sv, 'values'):
                    sv = sv.values
                sv = np.asarray(sv)
                if sv.ndim == 3:
                    sv = sv[:, :, 1]
                return np.asarray(sv[0], dtype=np.float64)
        except Exception as exc:
            logger.debug(f"Local SHAP calculation failed: {exc}")
        return None

    def _top_shap_features(self, feature_vector: np.ndarray, limit: int = 3) -> List[Dict]:
        """Return the top SHAP drivers for this transaction."""
        shap_values = self._compute_shap_values(feature_vector)
        if shap_values is None:
            return []

        top = sorted(
            zip(self.config.feature_columns, shap_values),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:limit]
        return [
            {'feature': feature, 'shap_value': round(float(value), 6)}
            for feature, value in top
        ]

    def predict(self, transaction: Dict) -> Dict:
        """Predict fraud for a single transaction."""
        with self._lock:
            feature_vector = self.feature_state.get_feature_vector(transaction)

        fraud_score = float(self.model.predict_proba(
            feature_vector.reshape(1, -1)
        )[0, 1])

        # Determine base anomaly
        base_anomaly = fraud_score >= self.config.anomaly_threshold

        # Sequential and geographic rules
        user_id = transaction.get(self.config.sender_col)
        ts_raw = transaction.get(self.config.timestamp_col)
        # Parse timestamp
        try:
            current_ts = pd.to_datetime(ts_raw) if ts_raw is not None else None
        except Exception:
            current_ts = None
        country = None
        if hasattr(self.config, 'sender_country_col') and self.config.sender_country_col:
            country = transaction.get(self.config.sender_country_col)

        # Geographic rule: different country within one hour
        geo_anomaly = False
        if user_id is not None and current_ts is not None:
            last_info = self.user_last_tx_info.get(user_id)
            if last_info:
                last_ts, last_country = last_info
                if last_country and country and last_country != country:
                    if (current_ts - last_ts).total_seconds() <= self.geo_window_seconds:
                        geo_anomaly = True

        # Sequential rule: flag if user had N consecutive anomalies
        seq_anomaly = False
        if user_id is not None:
            seq_anomaly = self.user_consecutive_anomalies.get(user_id, 0) >= self.sequential_threshold

        final_anomaly = base_anomaly or geo_anomaly or seq_anomaly

        # Update state counters
        if user_id is not None:
            self.user_consecutive_anomalies[user_id] = (
                    self.user_consecutive_anomalies.get(user_id, 0) + 1
            ) if final_anomaly else 0
            self.user_last_tx_info[user_id] = (current_ts, country)

        # update previous fraud count only after the current transaction
        # has been evaluated, so prev_fraud_count never includes itself.
        with self._lock:
            self.feature_state.update_fraud_count(transaction, final_anomaly)

        result = {
            'fraud_score': fraud_score,
            'is_anomaly': final_anomaly
        }

        # R13: only attach per-transaction SHAP for anomaly messages to keep
        # Kafka payloads small while still feeding the Grafana SHAP panel.
        if final_anomaly:
            top_shap = self._top_shap_features(feature_vector, limit=3)
            if top_shap:
                result['top_shap_features'] = top_shap

        return result

    # NEW SLOWER BUT CORRECT SEQUENTIAL WAY
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for a batch of transactions.

        Process sequentially so user-history features remain correct inside the
        batch. This is important for R5 because prev_fraud_count for transaction
        N must include prior fraud-labelled transactions from the same user,
        including earlier transactions from this batch.
        """
        predictions = []
        for tx in transactions:
            predictions.append(self.predict(tx))
        return predictions

    # def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
    #     """Predict fraud for a batch of transactions."""
    #     predictions = []
    #     with self._lock:
    #         feature_vectors = []
    #         # Process sequentially for proper feature state
    #         for tx in transactions:
    #             fv = self.feature_state.get_feature_vector(tx)
    #             feature_vectors.append(fv)
    #     # Batch prediction
    #     X = np.array(feature_vectors)
    #     fraud_scores = self.model.predict_proba(X)[:, 1]
    #
    #     for tx, score in zip(transactions, fraud_scores):
    #         fraud_score = float(score)
    #         base_anomaly = fraud_score >= self.config.anomaly_threshold
    #         user_id = tx.get(self.config.sender_col)
    #         ts_raw = tx.get(self.config.timestamp_col)
    #         try:
    #             current_ts = pd.to_datetime(ts_raw) if ts_raw is not None else None
    #         except Exception:
    #             current_ts = None
    #         country = None
    #         if hasattr(self.config, 'sender_country_col') and self.config.sender_country_col:
    #             country = tx.get(self.config.sender_country_col)
    #         geo_anomaly = False
    #         if user_id is not None and current_ts is not None:
    #             last_info = self.user_last_tx_info.get(user_id)
    #             if last_info:
    #                 last_ts, last_country = last_info
    #                 if last_country and country and last_country != country:
    #                     if (current_ts - last_ts).total_seconds() <= self.geo_window_seconds:
    #                         geo_anomaly = True
    #         seq_anomaly = False
    #         if user_id is not None:
    #             seq_anomaly = self.user_consecutive_anomalies.get(user_id, 0) >= self.sequential_threshold
    #         final_anomaly = base_anomaly or geo_anomaly or seq_anomaly
    #         if user_id is not None:
    #             self.user_consecutive_anomalies[user_id] = (
    #                 self.user_consecutive_anomalies.get(user_id, 0) + 1
    #             ) if final_anomaly else 0
    #             self.user_last_tx_info[user_id] = (current_ts, country)
    #         predictions.append({
    #             'fraud_score': fraud_score,
    #             'is_anomaly': final_anomaly
    #         })
    #
    #     return predictions


class APIFraudDetector:
    """
    Fraud detector using external API service.
    Use when scaling horizontally or need service isolation.
    """

    def __init__(
        self,
        api_url: str = 'http://localhost:8000',
        timeout: float = 5.0
    ):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Call batch prediction API."""
        try:
            response = self.session.post(
                f"{self.api_url}/predict/batch",
                json={"transactions": transactions},
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            predictions = []
            for p in result['predictions']:
                pred = {
                    'fraud_score': p['fraud_score'],
                    'is_anomaly': p['is_anomaly']
                }
                if p.get('top_shap_features'):
                    pred['top_shap_features'] = p['top_shap_features']
                predictions.append(pred)
            return predictions

        except Exception as e:
            logger.error(f"API prediction failed: {e}")
            # Fallback: mark all as non-anomaly
            return [{'fraud_score': 0.0, 'is_anomaly': False}] * len(transactions)


class StreamingPipeline:
    """
    Main streaming pipeline: consume -> predict -> route
    """

    def __init__(
        self,
        detector: FraudDetector,
        kafka_config: KafkaConfig = None,
        batch_size: int = 100
    ):
        self.detector = detector
        self.config = kafka_config or KafkaConfig.from_env()
        self.batch_size = batch_size
        self.consumer = None
        self.router = None

        # Metrics
        self.processed_count = 0
        self.anomaly_count = 0
        self.start_time = None
        self.last_log_time = None
        self.last_log_count = 0

    def start(self):
        """Initialize Kafka connections."""
        logger.info("Starting streaming pipeline...")

        self.consumer = TransactionConsumer(
            config=self.config,
            topics=[self.config.input_topic],
            auto_commit=False
        )

        self.router = TransactionRouter(self.config)

        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_count = 0

        return self

    def process_batch(self, transactions: List[Dict]) -> tuple:
        """Process a batch of transactions."""
        if not transactions:
            return 0, 0

        # Get predictions
        predictions = self.detector.predict_batch(transactions)

        # Route to appropriate topics
        normal, anomaly = self.router.route_batch(
            transactions,
            predictions,
            model_version=self.detector.model_version
        )

        return normal, anomaly

    def run(self, max_messages: int = None, log_interval: int = 5):
        """
        Run the streaming pipeline continuously.

        Args:
            max_messages: Stop after processing this many (None = infinite)
            log_interval: Seconds between progress logs
        """
        logger.info(f"Pipeline running. Batch size: {self.batch_size}")
        logger.info(f"Input topic: {self.config.input_topic}")
        logger.info(f"Output topics: {self.config.normal_topic}, {self.config.anomaly_topic}")

        try:
            while True:
                # Check if we've hit the limit
                if max_messages and self.processed_count >= max_messages:
                    logger.info(f"Reached max messages: {max_messages}")
                    break

                # Consume batch
                messages = self.consumer.consume_batch(
                    max_messages=self.batch_size,
                    timeout_ms=1000
                )

                if not messages:
                    continue

                # Extract transaction values
                transactions = [m['value'] for m in messages]

                # Process
                normal, anomaly = self.process_batch(transactions)

                self.processed_count += len(transactions)
                self.anomaly_count += anomaly

                # Commit offsets
                self.consumer.commit()

                # Periodic logging
                now = time.time()
                if now - self.last_log_time >= log_interval:
                    elapsed = now - self.start_time
                    interval_count = self.processed_count - self.last_log_count
                    interval_rate = interval_count / (now - self.last_log_time)
                    overall_rate = self.processed_count / elapsed

                    logger.info(
                        f"Processed: {self.processed_count:,} | "
                        f"Anomalies: {self.anomaly_count:,} ({self.anomaly_count/max(1,self.processed_count)*100:.2f}%) | "
                        f"Rate: {interval_rate:.0f} tx/sec (avg: {overall_rate:.0f})"
                    )

                    self.last_log_time = now
                    self.last_log_count = self.processed_count

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
        finally:
            self.stop()

    def stop(self):
        """Stop the pipeline and print summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        logger.info(f"\n{'='*50}")
        logger.info("Pipeline Summary")
        logger.info(f"{'='*50}")
        logger.info(f"Total processed: {self.processed_count:,}")
        logger.info(f"Total anomalies: {self.anomaly_count:,}")
        logger.info(f"Anomaly rate: {self.anomaly_count/max(1,self.processed_count)*100:.2f}%")
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Throughput: {self.processed_count/max(0.1,elapsed):.0f} tx/sec")

        if self.consumer:
            self.consumer.close()
        if self.router:
            self.router.close()


def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Consumer')
    parser.add_argument(
        '--mode',
        choices=['local', 'api'],
        default='local',
        help='Inference mode: local (in-process) or api (external service)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/fraud_model.joblib',
        help='Path to model (for local mode)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='models/feature_config.json',
        help='Path to feature config'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='API URL (for api mode)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--max-messages',
        type=int,
        default=None,
        help='Stop after N messages (default: run forever)'
    )
    parser.add_argument(
        '--kafka-servers',
        type=str,
        default=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        help='Kafka bootstrap servers'
    )

    args = parser.parse_args()

    # Configure Kafka
    kafka_config = KafkaConfig(
        bootstrap_servers=args.kafka_servers,
        batch_size=args.batch_size
    )

    # Initialize detector
    if args.mode == 'local':
        detector = FraudDetector(
            model_path=args.model,
            config_path=args.config
        ).load()
    else:
        detector = APIFraudDetector(api_url=args.api_url)

    # Create and run pipeline
    pipeline = StreamingPipeline(
        detector=detector,
        kafka_config=kafka_config,
        batch_size=args.batch_size
    )

    pipeline.start()
    pipeline.run(max_messages=args.max_messages)


if __name__ == '__main__':
    main()