# Financial Fraud Detection System

End-to-end ML system for real-time fraud detection in financial transactions with offline training, online streaming inference via Kafka, and monitoring dashboards.

## Architecture

```
                                    ┌─────────────────┐
                                    │   Training      │
                                    │   (Batch)       │
                                    └────────┬────────┘
                                             │
                                             ▼
┌──────────────┐    ┌─────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Transactions │───▶│   Kafka     │──▶│  Fraud Detector │──▶│  Kafka Topics   │
│   (Input)    │    │ (streaming) │   │  (Consumer)     │   │ normal/anomaly  │
└──────────────┘    └─────────────┘   └─────────────────┘   └────────┬────────┘
                                                                     │
                                    ┌────────────────────────────────┤
                                    │                                │
                                    ▼                                ▼
                           ┌─────────────────┐              ┌─────────────────┐
                           │   Monitoring    │              │   Downstream    │
                           │   Dashboard     │              │   Systems       │
                           └─────────────────┘              └─────────────────┘
```

## Features

- **Offline Training**: XGBoost/LightGBM model with class imbalance handling
- **Streaming Features**: Velocity features with bounded state (5m/1h/24h windows)
- **High Throughput**: Batched Kafka processing, ~100K+ tx/sec
- **Adaptive Columns**: Automatically detects and uses available dataset columns
- **Cost-Sensitive**: F2 score optimization (recall-weighted for fraud detection)
- **Monitoring**: Prometheus metrics + console dashboard

## Project Structure

```
Financial-fraud-detection/
├── src/
│   ├── training/
│   │   ├── features.py      # Feature engineering (batch + streaming)
│   │   ├── train.py         # Model training pipeline
│   │   └── evaluate.py      # Evaluation metrics & plots
│   ├── streaming/
│   │   ├── kafka_io.py      # Kafka producer/consumer utilities
│   │   ├── kafka_producer.py # Simulation producer
│   │   ├── consumer.py      # Main fraud detection consumer
│   │   └── predict_service/ # FastAPI inference service
│   └── monitoring/
│       └── metrics_consumer.py  # Prometheus metrics
├── models/                  # Model artifacts (generated)
├── docker-compose.yml       # Full stack deployment
├── Dockerfile.*            # Service containers
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Place your dataset CSV in the project root, then:

```bash
python -m src.training.train --data financial_fraud_detection_dataset.csv --output models
```

This will:
- Perform EDA and feature engineering
- Split data (60% train, 20% test, 20% holdout for streaming)
- Train XGBoost model with class imbalance handling
- Find optimal threshold (F2-optimized)
- Save `models/fraud_model.joblib` and `models/feature_config.json`
- Save holdout set for streaming simulation

### 3. Start Streaming Infrastructure

```bash
# Start Kafka + Zookeeper
docker-compose up -d kafka zookeeper kafka-init

# Wait for Kafka to be ready
docker-compose logs -f kafka-init
```

### 4. Run Fraud Detection Consumer

```bash
# Option A: Run locally
python -m src.streaming.consumer --mode local

# Option B: Run in Docker
docker-compose up -d fraud-consumer
```

### 5. Simulate Transactions

```bash
# Option A: Run locally
python -m src.streaming.kafka_producer --data models/data/streaming_holdout.csv

# Option B: Run in Docker (burst test)
docker-compose --profile simulation up simulator
```

### 6. Monitor Results

```bash
# Option A: Run locally with console dashboard
python -m src.monitoring.metrics_consumer

# Option B: Run in Docker
docker-compose up -d monitoring

# Access metrics at http://localhost:9090/metrics
```

## Full Docker Deployment

```bash
# Build and start everything
docker-compose up -d

# Run simulation (after services are ready)
docker-compose --profile simulation up simulator

# Optional: Add Kafka UI for debugging
docker-compose --profile debug up -d kafka-ui
# Access at http://localhost:8080

# Optional: Add Prometheus + Grafana
docker-compose --profile observability up -d
# Prometheus: http://localhost:9091
# Grafana: http://localhost:3000 (admin/admin)
```

## Training Options

```bash
python -m src.training.train \
    --data your_dataset.csv \
    --output models \
    --model-type xgboost \           # or lightgbm, random_forest
    --threshold-method f2 \          # or cost, recall_at_precision
    --min-precision 0.5              # for recall_at_precision method
```

## Streaming Consumer Options

```bash
python -m src.streaming.consumer \
    --mode local \                   # or api (for external service)
    --model models/fraud_model.joblib \
    --config models/feature_config.json \
    --batch-size 100 \
    --kafka-servers localhost:9092
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker addresses |
| `KAFKA_INPUT_TOPIC` | `transactions` | Input topic name |
| `KAFKA_NORMAL_TOPIC` | `normal_transactions` | Normal output topic |
| `KAFKA_ANOMALY_TOPIC` | `anomaly_transactions` | Anomaly output topic |
| `KAFKA_BATCH_SIZE` | `100` | Batch size for processing |
| `MODEL_PATH` | `models/fraud_model.joblib` | Model file path |
| `CONFIG_PATH` | `models/feature_config.json` | Config file path |

## Feature Engineering

### Batch Features (Training)
- **DateTime**: hour, day, day_of_week, week, month, is_weekend, is_night
- **User Aggregates**: avg/std/min/max amount, transaction count
- **Velocity Windows**: count, sum, avg in 5m/1h/24h windows
- **Ratios**: amount_to_user_avg, amount_zscore, amount_to_rolling_avg
- **Cross-border**: is_cross_border (if country columns exist)
- **Encoded Categoricals**: transaction_type, country codes

### Streaming Features
Uses bounded state per user:
- Max 20 transactions history
- 24h TTL for old transactions
- Same velocity calculations as batch

## Performance

Target throughput: **~100K+ transactions in ~5 seconds**

Achieved through:
- Batched Kafka consumption/production
- In-process model inference (no API calls per transaction)
- LZ4 compression for Kafka messages
- Bounded user state (no memory leaks)

## API Endpoints (Predict Service)

When running the FastAPI predict service:

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "tx1", "timestamp": "2024-01-01T12:00:00", "sender_account": "acc1", "amount": 1000.0}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'

# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Output Schema

Messages in normal_transactions / anomaly_transactions topics:

```json
{
  "transaction_id": "tx123",
  "timestamp": "2024-01-01T12:00:00",
  "sender_account": "acc456",
  "amount": 5000.0,
  "fraud_score": 0.85,
  "is_anomaly": true,
  "model_version": "1.0",
  "processed_at": "2024-01-01T12:00:01.123Z"
}
```

## Monitoring Metrics

Prometheus metrics available at `/metrics`:

- `fraud_anomalies_total` - Total anomalies detected
- `fraud_normal_total` - Total normal transactions
- `fraud_anomaly_rate` - Anomalies per second (windowed)
- `fraud_avg_anomaly_amount` - Average anomaly transaction amount
- `fraud_score_bucket` - Score distribution histogram

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## Troubleshooting

### Kafka Connection Issues
```bash
# Check Kafka is running
docker-compose ps
docker-compose logs kafka

# Test connectivity
kafka-topics --bootstrap-server localhost:9092 --list
```

### Model Not Found
```bash
# Ensure model is trained
python -m src.training.train --data your_data.csv

# Check model files exist
ls -la models/
```

### Out of Memory
- Reduce `KAFKA_BATCH_SIZE`
- Ensure bounded state is working (check `max_history_per_user`)
- Increase container memory limits in docker-compose.yml

## License

MIT
