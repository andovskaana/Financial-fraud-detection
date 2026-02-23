"""
Kafka Producer for Streaming Simulation

Reads the holdout dataset and produces transactions to Kafka
at configurable rates to simulate real-time streaming.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.streaming.kafka_io import TransactionProducer, KafkaConfig, create_topics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransactionSimulator:
    """
    Simulates real-time transaction stream from holdout dataset.
    """

    def __init__(
        self,
        data_path: str,
        kafka_config: KafkaConfig = None,
        batch_size: int = 1000,
        rate_limit: int = 0  # 0 = no limit (max speed)
    ):
        self.data_path = data_path
        self.config = kafka_config or KafkaConfig.from_env()
        self.batch_size = batch_size
        self.rate_limit = rate_limit  # transactions per second
        self.producer = None
        self.df = None

    def load_data(self):
        """Load holdout dataset."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df):,} transactions")
        return self

    def _prepare_transaction(self, row: pd.Series) -> dict:
        """Convert dataframe row to transaction dict."""
        tx = row.to_dict()

        # Ensure timestamp is string (ISO format)
        if 'timestamp' in tx:
            if pd.notna(tx['timestamp']):
                if isinstance(tx['timestamp'], str):
                    pass  # Already string
                else:
                    tx['timestamp'] = pd.Timestamp(tx['timestamp']).isoformat()
            else:
                tx['timestamp'] = datetime.utcnow().isoformat()
        else:
            tx['timestamp'] = datetime.utcnow().isoformat()

        # Handle NaN values
        for key, value in tx.items():
            if pd.isna(value):
                tx[key] = None

        return tx

    def run(
        self,
        limit: int = None,
        progress_interval: int = 10000
    ):
        """
        Run the simulation, producing transactions to Kafka.

        Args:
            limit: Maximum number of transactions to send (None = all)
            progress_interval: Print progress every N transactions
        """
        if self.df is None:
            self.load_data()

        # Create topics
        create_topics(
            self.config.bootstrap_servers,
            [self.config.input_topic, self.config.normal_topic, self.config.anomaly_topic]
        )

        # Initialize producer
        self.producer = TransactionProducer(self.config)

        # Determine how many transactions to send
        total = len(self.df) if limit is None else min(limit, len(self.df))

        logger.info(f"Starting simulation: {total:,} transactions")
        logger.info(f"Target topic: {self.config.input_topic}")
        logger.info(f"Rate limit: {self.rate_limit} tx/sec" if self.rate_limit > 0 else "Rate limit: MAX SPEED")

        start_time = time.time()
        sent_count = 0
        batch = []

        try:
            for idx, row in self.df.iterrows():
                if limit and sent_count >= limit:
                    break

                tx = self._prepare_transaction(row)
                batch.append(tx)

                # Send batch when full
                if len(batch) >= self.batch_size:
                    self._send_batch(batch)
                    sent_count += len(batch)
                    batch = []

                    # Progress logging
                    if sent_count % progress_interval == 0:
                        elapsed = time.time() - start_time
                        rate = sent_count / elapsed
                        logger.info(
                            f"Progress: {sent_count:,}/{total:,} "
                            f"({sent_count/total*100:.1f}%) - "
                            f"{rate:.0f} tx/sec"
                        )

                    # Rate limiting
                    if self.rate_limit > 0:
                        target_time = sent_count / self.rate_limit
                        actual_time = time.time() - start_time
                        if actual_time < target_time:
                            time.sleep(target_time - actual_time)

            # Send remaining batch
            if batch:
                self._send_batch(batch)
                sent_count += len(batch)

            # Flush and finalize
            self.producer.flush()
            elapsed = time.time() - start_time

            logger.info(f"\n{'='*50}")
            logger.info(f"Simulation Complete!")
            logger.info(f"{'='*50}")
            logger.info(f"Total sent: {sent_count:,} transactions")
            logger.info(f"Elapsed time: {elapsed:.2f} seconds")
            logger.info(f"Average rate: {sent_count/elapsed:.0f} tx/sec")

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self.producer.close()

        return sent_count

    def _send_batch(self, batch: list):
        """Send a batch of transactions."""
        for tx in batch:
            # Use sender_account as partition key for ordering
            key = tx.get('sender_account') or tx.get('sender')
            self.producer.send(
                self.config.input_topic,
                tx,
                key=str(key) if key else None
            )

    def run_burst(
        self,
        num_transactions: int = 100000,
        target_duration_sec: float = 5.0
    ):
        """
        Run a burst test: send N transactions as fast as possible.

        Args:
            num_transactions: Number of transactions to send
            target_duration_sec: Target duration (just for reference)
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"BURST TEST: {num_transactions:,} transactions")
        logger.info(f"{'='*50}")

        self.rate_limit = 0  # No rate limit
        self.batch_size = 5000  # Large batches for throughput

        start = time.time()
        sent = self.run(limit=num_transactions, progress_interval=10000)
        elapsed = time.time() - start

        logger.info(f"\nBurst Results:")
        logger.info(f"  Transactions: {sent:,}")
        logger.info(f"  Duration: {elapsed:.2f} sec")
        logger.info(f"  Throughput: {sent/elapsed:,.0f} tx/sec")

        return sent, elapsed


def main():
    parser = argparse.ArgumentParser(description='Transaction Stream Simulator')
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='models/data/streaming_holdout.csv',
        help='Path to holdout dataset CSV'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum transactions to send (default: all)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1000,
        help='Batch size for Kafka producer'
    )
    parser.add_argument(
        '--rate', '-r',
        type=int,
        default=0,
        help='Rate limit (tx/sec, 0=unlimited)'
    )
    parser.add_argument(
        '--burst',
        action='store_true',
        help='Run burst test mode'
    )
    parser.add_argument(
        '--burst-count',
        type=int,
        default=100000,
        help='Number of transactions for burst test'
    )
    parser.add_argument(
        '--kafka-servers',
        type=str,
        default=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default=os.getenv('KAFKA_INPUT_TOPIC', 'transactions'),
        help='Input topic name'
    )

    args = parser.parse_args()

    # Configure Kafka
    config = KafkaConfig(
        bootstrap_servers=args.kafka_servers,
        input_topic=args.topic,
        batch_size=args.batch_size
    )

    # Create simulator
    simulator = TransactionSimulator(
        data_path=args.data,
        kafka_config=config,
        batch_size=args.batch_size,
        rate_limit=args.rate
    )

    # Load data
    simulator.load_data()

    # Run simulation
    if args.burst:
        simulator.run_burst(num_transactions=args.burst_count)
    else:
        simulator.run(limit=args.limit)


if __name__ == '__main__':
    main()
