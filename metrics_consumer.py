"""
Metrics Consumer for Fraud Detection Monitoring

Consumes from anomaly topic and exposes Prometheus metrics.
Also provides a simple Streamlit dashboard option.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from threading import Thread, Lock
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.streaming.kafka_io import TransactionConsumer, KafkaConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and aggregates metrics from the anomaly topic.
    """

    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        self._lock = Lock()

        # Metrics
        self.total_anomalies = 0
        self.total_normal = 0

        # Time-windowed metrics (last N seconds)
        self.anomaly_timestamps = deque()
        self.normal_timestamps = deque()

        # Score distribution
        self.score_buckets = defaultdict(int)  # 0.0-0.1, 0.1-0.2, etc.

        # Top senders by anomaly count
        self.sender_anomaly_counts = defaultdict(int)

        # Amount statistics
        self.anomaly_amounts = deque(maxlen=1000)

        # Processing stats
        self.start_time = datetime.utcnow()
        self.last_message_time = None

    def _cleanup_window(self, timestamps: deque):
        """Remove timestamps outside the window."""
        cutoff = time.time() - self.window_size
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

    def record_anomaly(self, message: dict):
        """Record an anomaly transaction."""
        with self._lock:
            now = time.time()
            self.total_anomalies += 1
            self.anomaly_timestamps.append(now)
            self._cleanup_window(self.anomaly_timestamps)

            # Score bucket
            score = message.get('fraud_score', 0.0)
            bucket = min(int(score * 10), 9)  # 0-9
            self.score_buckets[bucket] += 1

            # Sender tracking
            sender = message.get('sender_account', message.get('sender', 'unknown'))
            self.sender_anomaly_counts[sender] += 1

            # Amount
            amount = message.get('amount', 0)
            self.anomaly_amounts.append(amount)

            self.last_message_time = datetime.utcnow()

    def record_normal(self, message: dict):
        """Record a normal transaction."""
        with self._lock:
            now = time.time()
            self.total_normal += 1
            self.normal_timestamps.append(now)
            self._cleanup_window(self.normal_timestamps)
            self.last_message_time = datetime.utcnow()

    def get_metrics(self) -> dict:
        """Get current metrics snapshot."""
        with self._lock:
            self._cleanup_window(self.anomaly_timestamps)
            self._cleanup_window(self.normal_timestamps)

            uptime = (datetime.utcnow() - self.start_time).total_seconds()

            # Calculate rates
            anomaly_rate = len(self.anomaly_timestamps) / self.window_size if self.window_size > 0 else 0
            normal_rate = len(self.normal_timestamps) / self.window_size if self.window_size > 0 else 0

            # Top senders
            top_senders = sorted(
                self.sender_anomaly_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            # Amount stats
            amounts = list(self.anomaly_amounts)
            avg_amount = sum(amounts) / len(amounts) if amounts else 0
            max_amount = max(amounts) if amounts else 0

            return {
                'total_anomalies': self.total_anomalies,
                'total_normal': self.total_normal,
                'total_processed': self.total_anomalies + self.total_normal,
                'anomaly_rate_per_sec': anomaly_rate,
                'normal_rate_per_sec': normal_rate,
                'anomalies_in_window': len(self.anomaly_timestamps),
                'window_seconds': self.window_size,
                'score_distribution': dict(self.score_buckets),
                'top_senders': top_senders,
                'avg_anomaly_amount': avg_amount,
                'max_anomaly_amount': max_amount,
                'uptime_seconds': uptime,
                'last_message': self.last_message_time.isoformat() if self.last_message_time else None
            }

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output."""
        metrics = self.get_metrics()
        lines = []

        # Counters
        lines.append(f'fraud_anomalies_total {metrics["total_anomalies"]}')
        lines.append(f'fraud_normal_total {metrics["total_normal"]}')
        lines.append(f'fraud_processed_total {metrics["total_processed"]}')

        # Gauges
        lines.append(f'fraud_anomaly_rate {metrics["anomaly_rate_per_sec"]:.4f}')
        lines.append(f'fraud_normal_rate {metrics["normal_rate_per_sec"]:.4f}')
        lines.append(f'fraud_anomalies_in_window {metrics["anomalies_in_window"]}')
        lines.append(f'fraud_avg_anomaly_amount {metrics["avg_anomaly_amount"]:.2f}')
        lines.append(f'fraud_uptime_seconds {metrics["uptime_seconds"]:.2f}')

        # Score histogram
        for bucket, count in metrics['score_distribution'].items():
            lines.append(f'fraud_score_bucket{{le="{(bucket+1)/10:.1f}"}} {count}')

        return '\n'.join(lines)


def run_prometheus_server(collector: MetricsCollector, port: int = 9090):
    """Run a simple HTTP server for Prometheus scraping."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/metrics':
                metrics = collector.get_prometheus_metrics()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(metrics.encode())
            elif self.path == '/health':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "healthy"}')
            elif self.path == '/stats':
                metrics = collector.get_metrics()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(metrics, indent=2).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress logging

    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    logger.info(f"Prometheus metrics server on http://0.0.0.0:{port}/metrics")
    server.serve_forever()


def run_console_dashboard(collector: MetricsCollector, refresh_interval: int = 2):
    """Run a simple console-based dashboard."""
    try:
        while True:
            metrics = collector.get_metrics()

            # Clear screen (works on most terminals)
            print('\033[2J\033[H', end='')

            print("=" * 60)
            print(" FRAUD DETECTION MONITORING DASHBOARD")
            print("=" * 60)
            print()

            print(f"Uptime: {metrics['uptime_seconds']:.0f} seconds")
            print(f"Last message: {metrics['last_message'] or 'None'}")
            print()

            print("-" * 40)
            print(" Transaction Counts")
            print("-" * 40)
            print(f"  Total Processed:  {metrics['total_processed']:,}")
            print(f"  Normal:           {metrics['total_normal']:,}")
            print(f"  Anomalies:        {metrics['total_anomalies']:,}")

            if metrics['total_processed'] > 0:
                anomaly_pct = metrics['total_anomalies'] / metrics['total_processed'] * 100
                print(f"  Anomaly Rate:     {anomaly_pct:.2f}%")
            print()

            print("-" * 40)
            print(f" Rates (last {metrics['window_seconds']}s)")
            print("-" * 40)
            print(f"  Anomalies/sec:    {metrics['anomaly_rate_per_sec']:.2f}")
            print(f"  Normal/sec:       {metrics['normal_rate_per_sec']:.2f}")
            print(f"  Anomalies in window: {metrics['anomalies_in_window']}")
            print()

            print("-" * 40)
            print(" Anomaly Amount Stats")
            print("-" * 40)
            print(f"  Average:  ${metrics['avg_anomaly_amount']:,.2f}")
            print(f"  Maximum:  ${metrics['max_anomaly_amount']:,.2f}")
            print()

            print("-" * 40)
            print(" Top Suspicious Senders")
            print("-" * 40)
            for i, (sender, count) in enumerate(metrics['top_senders'][:5], 1):
                print(f"  {i}. {sender[:30]:30} - {count} anomalies")
            print()

            print("-" * 40)
            print(" Score Distribution")
            print("-" * 40)
            for bucket in range(10):
                count = metrics['score_distribution'].get(bucket, 0)
                bar = '#' * min(count // 10, 30)
                print(f"  {bucket/10:.1f}-{(bucket+1)/10:.1f}: {bar} ({count})")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\nDashboard stopped")


class MonitoringPipeline:
    """
    Main monitoring pipeline.
    """

    def __init__(
        self,
        kafka_config: KafkaConfig = None,
        topics: list = None,
        metrics_port: int = 9090,
        enable_console: bool = True
    ):
        self.config = kafka_config or KafkaConfig.from_env()
        self.topics = topics or [
            self.config.anomaly_topic,
            self.config.normal_topic
        ]
        self.metrics_port = metrics_port
        self.enable_console = enable_console

        self.collector = MetricsCollector()
        self.consumer = None

    def start(self):
        """Start the monitoring pipeline."""
        logger.info(f"Starting monitoring for topics: {self.topics}")

        # Start Prometheus metrics server
        metrics_thread = Thread(
            target=run_prometheus_server,
            args=(self.collector, self.metrics_port),
            daemon=True
        )
        metrics_thread.start()

        # Initialize consumer
        self.consumer = TransactionConsumer(
            config=self.config,
            topics=self.topics,
            auto_commit=True
        )

        return self

    def run(self):
        """Run the monitoring loop."""
        # Start console dashboard in background if enabled
        if self.enable_console:
            dashboard_thread = Thread(
                target=run_console_dashboard,
                args=(self.collector, 2),
                daemon=True
            )
            dashboard_thread.start()

        logger.info("Monitoring running. Press Ctrl+C to stop.")

        try:
            while True:
                messages = self.consumer.consume_batch(
                    max_messages=100,
                    timeout_ms=1000
                )

                for msg in messages:
                    value = msg['value']
                    topic = msg['topic']

                    if 'anomaly' in topic:
                        self.collector.record_anomaly(value)
                    else:
                        self.collector.record_normal(value)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
        finally:
            if self.consumer:
                self.consumer.close()


def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Monitoring')
    parser.add_argument(
        '--kafka-servers',
        type=str,
        default=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--anomaly-topic',
        type=str,
        default='anomaly_transactions',
        help='Anomaly topic name'
    )
    parser.add_argument(
        '--normal-topic',
        type=str,
        default='normal_transactions',
        help='Normal topic name'
    )
    parser.add_argument(
        '--metrics-port',
        type=int,
        default=9090,
        help='Prometheus metrics port'
    )
    parser.add_argument(
        '--no-console',
        action='store_true',
        help='Disable console dashboard'
    )

    args = parser.parse_args()

    # Configure
    kafka_config = KafkaConfig(
        bootstrap_servers=args.kafka_servers,
        anomaly_topic=args.anomaly_topic,
        normal_topic=args.normal_topic
    )

    topics = [args.anomaly_topic, args.normal_topic]

    # Run
    pipeline = MonitoringPipeline(
        kafka_config=kafka_config,
        topics=topics,
        metrics_port=args.metrics_port,
        enable_console=not args.no_console
    )

    pipeline.start()
    pipeline.run()


if __name__ == '__main__':
    main()
