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

    def load(self):
        """Load model and initialize feature state."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info(f"Loading config from {self.config_path}")
        self.config = FeatureConfig.load(self.config_path)

        self.feature_state = StreamingFeatureState(self.config)

        logger.info(f"Detector ready. Threshold: {self.config.anomaly_threshold}")
        return self

    def predict(self, transaction: Dict) -> Dict:
        """Predict fraud for a single transaction."""
        with self._lock:
            feature_vector = self.feature_state.get_feature_vector(transaction)

        fraud_score = float(self.model.predict_proba(
            feature_vector.reshape(1, -1)
        )[0, 1])

        is_anomaly = fraud_score >= self.config.anomaly_threshold

        return {
            'fraud_score': fraud_score,
            'is_anomaly': is_anomaly
        }

    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for a batch of transactions."""
        predictions = []

        with self._lock:
            # Process sequentially for proper feature state
            feature_vectors = []
            for tx in transactions:
                fv = self.feature_state.get_feature_vector(tx)
                feature_vectors.append(fv)

        # Batch prediction
        X = np.array(feature_vectors)
        fraud_scores = self.model.predict_proba(X)[:, 1]

        for score in fraud_scores:
            predictions.append({
                'fraud_score': float(score),
                'is_anomaly': float(score) >= self.config.anomaly_threshold
            })

        return predictions


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
            return [
                {
                    'fraud_score': p['fraud_score'],
                    'is_anomaly': p['is_anomaly']
                }
                for p in result['predictions']
            ]

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
