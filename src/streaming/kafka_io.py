"""
Kafka I/O Utilities for Fraud Detection Streaming Pipeline

Provides:
- Producer/Consumer wrappers
- Batch processing for high throughput
- Error handling and retries
"""

import os
import json
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Kafka configuration from environment variables."""
    bootstrap_servers: str = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    input_topic: str = os.getenv('KAFKA_INPUT_TOPIC', 'transactions')
    normal_topic: str = os.getenv('KAFKA_NORMAL_TOPIC', 'normal_transactions')
    anomaly_topic: str = os.getenv('KAFKA_ANOMALY_TOPIC', 'anomaly_transactions')
    consumer_group: str = os.getenv('KAFKA_CONSUMER_GROUP', 'fraud-detector')
    batch_size: int = int(os.getenv('KAFKA_BATCH_SIZE', '100'))
    batch_timeout_ms: int = int(os.getenv('KAFKA_BATCH_TIMEOUT_MS', '1000'))

    @classmethod
    def from_env(cls) -> 'KafkaConfig':
        return cls()


class TransactionProducer:
    """
    High-throughput Kafka producer for transactions.
    Uses batching and compression for performance.
    """

    def __init__(self, config: KafkaConfig = None):
        self.config = config or KafkaConfig.from_env()
        self.producer = None
        self._connect()

    def _connect(self, max_retries: int = 5):
        """Connect to Kafka with retry logic."""
        for attempt in range(max_retries):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.config.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    # Batching for high throughput
                    batch_size=16384,  # 16KB
                    linger_ms=self.config.batch_timeout_ms,
                    # Compression
                    compression_type='lz4',
                    # Reliability
                    acks='all',
                    retries=3,
                    # Buffer
                    buffer_memory=33554432,  # 32MB
                )
                logger.info(f"Connected to Kafka: {self.config.bootstrap_servers}")
                return
            except NoBrokersAvailable:
                logger.warning(f"Kafka not available, attempt {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise ConnectionError("Failed to connect to Kafka after retries")

    def send(
        self,
        topic: str,
        message: Dict,
        key: str = None,
        callback: Callable = None
    ):
        """Send a single message."""
        future = self.producer.send(topic, value=message, key=key)

        if callback:
            future.add_callback(callback)
            future.add_errback(lambda e: logger.error(f"Send failed: {e}"))

        return future

    def send_batch(
        self,
        topic: str,
        messages: List[Dict],
        key_field: str = None
    ) -> int:
        """
        Send a batch of messages efficiently.

        Args:
            topic: Target topic
            messages: List of message dictionaries
            key_field: Field to use as partition key (for ordering)

        Returns:
            Number of messages sent
        """
        sent = 0
        for msg in messages:
            key = str(msg.get(key_field)) if key_field and key_field in msg else None
            try:
                self.send(topic, msg, key=key)
                sent += 1
            except Exception as e:
                logger.error(f"Failed to send message: {e}")

        return sent

    def flush(self, timeout: float = 30.0):
        """Flush all pending messages."""
        self.producer.flush(timeout=timeout)

    def close(self):
        """Close the producer."""
        if self.producer:
            self.producer.close()


class TransactionConsumer:
    """
    Kafka consumer for transactions with batch processing.
    """

    def __init__(
        self,
        config: KafkaConfig = None,
        topics: List[str] = None,
        auto_commit: bool = False
    ):
        self.config = config or KafkaConfig.from_env()
        self.topics = topics or [self.config.input_topic]
        self.consumer = None
        self.auto_commit = auto_commit
        self._connect()

    def _connect(self, max_retries: int = 5):
        """Connect to Kafka with retry logic."""
        for attempt in range(max_retries):
            try:
                self.consumer = KafkaConsumer(
                    *self.topics,
                    bootstrap_servers=self.config.bootstrap_servers,
                    group_id=self.config.consumer_group,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    key_deserializer=lambda k: k.decode('utf-8') if k else None,
                    # Performance
                    fetch_min_bytes=1,
                    fetch_max_wait_ms=500,
                    max_poll_records=self.config.batch_size,
                    # Offset management
                    auto_offset_reset='earliest',
                    enable_auto_commit=self.auto_commit,
                )
                logger.info(f"Consumer connected to topics: {self.topics}")
                return
            except NoBrokersAvailable:
                logger.warning(f"Kafka not available, attempt {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)

        raise ConnectionError("Failed to connect to Kafka after retries")

    def consume_batch(
        self,
        max_messages: int = None,
        timeout_ms: int = 1000
    ) -> List[Dict]:
        """
        Consume a batch of messages.

        Args:
            max_messages: Maximum messages to return (default: config.batch_size)
            timeout_ms: Timeout for polling

        Returns:
            List of message value dictionaries
        """
        max_messages = max_messages or self.config.batch_size
        messages = []

        # Poll for messages
        records = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_messages)

        for topic_partition, partition_records in records.items():
            for record in partition_records:
                messages.append({
                    'value': record.value,
                    'key': record.key,
                    'topic': record.topic,
                    'partition': record.partition,
                    'offset': record.offset,
                    'timestamp': record.timestamp
                })

        return messages

    def consume_stream(
        self,
        handler: Callable[[Dict], Any],
        batch_handler: Callable[[List[Dict]], Any] = None,
        batch_size: int = None
    ):
        """
        Continuously consume messages with a handler.

        Args:
            handler: Function to process each message
            batch_handler: Optional function to process batches (more efficient)
            batch_size: Batch size for batch_handler
        """
        batch_size = batch_size or self.config.batch_size

        logger.info("Starting consumer stream...")

        try:
            while True:
                messages = self.consume_batch(max_messages=batch_size)

                if not messages:
                    continue

                if batch_handler:
                    # Process as batch
                    batch_handler([m['value'] for m in messages])
                else:
                    # Process individually
                    for msg in messages:
                        handler(msg['value'])

                # Manual commit after processing
                if not self.auto_commit:
                    self.consumer.commit()

        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self.close()

    def commit(self):
        """Manually commit offsets."""
        self.consumer.commit()

    def close(self):
        """Close the consumer."""
        if self.consumer:
            self.consumer.close()


class TransactionRouter:
    """
    Routes transactions to normal or anomaly topics based on prediction.
    """

    def __init__(self, config: KafkaConfig = None):
        self.config = config or KafkaConfig.from_env()
        self.producer = TransactionProducer(config)

    def route(
        self,
        transaction: Dict,
        is_anomaly: bool,
        fraud_score: float,
        model_version: str = "1.0"
    ):
        """
        Route a transaction to the appropriate topic.

        Args:
            transaction: Original transaction data
            is_anomaly: Whether transaction is flagged as anomaly
            fraud_score: Model's fraud probability score
            model_version: Version of the model used
        """
        # Enrich transaction with prediction
        enriched = {
            **transaction,
            'fraud_score': float(fraud_score),
            'is_anomaly': bool(is_anomaly),
            'model_version': model_version,
            'processed_at': datetime.utcnow().isoformat()
        }

        # Route based on prediction
        topic = self.config.anomaly_topic if is_anomaly else self.config.normal_topic

        # Use sender as partition key for ordering
        key = transaction.get('sender_account', transaction.get('sender'))

        self.producer.send(topic, enriched, key=str(key) if key else None)

    def route_batch(
        self,
        transactions: List[Dict],
        predictions: List[Dict],
        model_version: str = "1.0"
    ) -> tuple:
        """
        Route a batch of transactions.

        Args:
            transactions: List of original transactions
            predictions: List of dicts with 'fraud_score' and 'is_anomaly'
            model_version: Version of the model

        Returns:
            Tuple of (normal_count, anomaly_count)
        """
        normal_count = 0
        anomaly_count = 0

        for tx, pred in zip(transactions, predictions):
            is_anomaly = pred['is_anomaly']
            fraud_score = pred['fraud_score']

            self.route(tx, is_anomaly, fraud_score, model_version)

            if is_anomaly:
                anomaly_count += 1
            else:
                normal_count += 1

        self.producer.flush()

        return normal_count, anomaly_count

    def close(self):
        """Close the router."""
        self.producer.close()


def create_topics(
    bootstrap_servers: str = 'localhost:9092',
    topics: List[str] = None,
    num_partitions: int = 3,
    replication_factor: int = 1
):
    """
    Create Kafka topics if they don't exist.

    Note: In production, topics should be created via Kafka admin tools
    with proper configuration.
    """
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    topics = topics or ['transactions', 'normal_transactions', 'anomaly_transactions']

    try:
        admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)

        topic_list = [
            NewTopic(
                name=topic,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )
            for topic in topics
        ]

        admin.create_topics(new_topics=topic_list, validate_only=False)
        logger.info(f"Created topics: {topics}")

    except TopicAlreadyExistsError:
        logger.info("Topics already exist")
    except Exception as e:
        logger.warning(f"Could not create topics: {e}")
