"""
FastAPI Predict Service for Fraud Detection

High-throughput inference service with:
- Batched predictions
- In-memory model loading
- Streaming feature computation
- Health checks and metrics
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import joblib
import pandas as pd
from collections import defaultdict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.training.features import FeatureConfig, StreamingFeatureState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'models/fraud_model.joblib')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'models/feature_config.json')
MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0')

# Global state
class ModelState:
    model = None
    config: FeatureConfig = None
    feature_state: StreamingFeatureState = None
    is_loaded: bool = False
    load_time: datetime = None
    prediction_count: int = 0
    error_count: int = 0

    # State for sequential and geographic rules
    user_consecutive_anomalies: Dict[str, int] = {}
    user_last_tx_info: Dict[str, tuple] = {}
    sequential_threshold: int = 2
    geo_window_seconds: int = 3600


state = ModelState()

# Request/Response models
class Transaction(BaseModel):
    transaction_id: str
    timestamp: str
    sender_account: str
    amount: float
    receiver_account: Optional[str] = None
    transaction_type: Optional[str] = None
    sender_country: Optional[str] = None
    receiver_country: Optional[str] = None

    # Allow additional fields
    class Config:
        extra = 'allow'


class TransactionBatch(BaseModel):
    transactions: List[Transaction]


class PredictionResult(BaseModel):
    transaction_id: str
    fraud_score: float
    is_anomaly: bool
    model_version: str
    processing_time_ms: float


class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    total_transactions: int
    anomaly_count: int
    processing_time_ms: float
    throughput_tps: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    prediction_count: int
    error_count: int


def load_model():
    """Load model and configuration."""
    logger.info("Loading model and configuration...")

    try:
        # Load model
        state.model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")

        # Load config
        state.config = FeatureConfig.load(CONFIG_PATH)
        logger.info(f"Config loaded from {CONFIG_PATH}")

        # Initialize streaming feature state
        state.feature_state = StreamingFeatureState(state.config)
        logger.info("Feature state initialized")

        # Initialize rule state
        state.user_consecutive_anomalies = defaultdict(int)
        state.user_last_tx_info = {}
        state.sequential_threshold = 2
        state.geo_window_seconds = 3600

        state.is_loaded = True
        state.load_time = datetime.utcnow()

        logger.info(f"Model ready! Threshold: {state.config.anomaly_threshold}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection Service",
    description="Real-time fraud detection with streaming features",
    version=MODEL_VERSION,
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = 0.0
    if state.load_time:
        uptime = (datetime.utcnow() - state.load_time).total_seconds()

    return HealthResponse(
        status="healthy" if state.is_loaded else "unhealthy",
        model_loaded=state.is_loaded,
        model_version=MODEL_VERSION,
        uptime_seconds=uptime,
        prediction_count=state.prediction_count,
        error_count=state.error_count
    )


@app.post("/predict", response_model=PredictionResult)
async def predict_single(transaction: Transaction):
    """Predict fraud for a single transaction."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Convert to dict
        tx_dict = transaction.model_dump()

        # Compute features using streaming state
        feature_vector = state.feature_state.get_feature_vector(tx_dict)

        # Predict
        fraud_score = float(state.model.predict_proba(
            feature_vector.reshape(1, -1)
        )[0, 1])

        # Determine base anomaly based on model threshold
        base_anomaly = fraud_score >= state.config.anomaly_threshold

        # Sequential and geographic rules
        user_id = tx_dict.get(state.config.sender_col)
        ts_raw = tx_dict.get(state.config.timestamp_col)
        country = None
        if hasattr(state.config, 'sender_country_col') and state.config.sender_country_col:
            country = tx_dict.get(state.config.sender_country_col)
        # Parse timestamp
        try:
            current_ts = pd.to_datetime(ts_raw) if ts_raw is not None else None
        except Exception:
            current_ts = None

        # Geographic rule
        geo_anomaly = False
        if user_id is not None and current_ts is not None:
            last_info = state.user_last_tx_info.get(user_id)
            if last_info:
                last_ts, last_country = last_info
                if last_country and country and last_country != country:
                    if (current_ts - last_ts).total_seconds() <= state.geo_window_seconds:
                        geo_anomaly = True

        # Sequential rule
        seq_anomaly = False
        if user_id is not None:
            seq_anomaly = state.user_consecutive_anomalies.get(user_id, 0) >= state.sequential_threshold

        final_anomaly = base_anomaly or geo_anomaly or seq_anomaly

        # Update state counters
        if user_id is not None:
            state.user_consecutive_anomalies[user_id] = (
                state.user_consecutive_anomalies.get(user_id, 0) + 1
            ) if final_anomaly else 0
            state.user_last_tx_info[user_id] = (current_ts, country)

        processing_time = (time.time() - start_time) * 1000
        state.prediction_count += 1

        return PredictionResult(
            transaction_id=transaction.transaction_id,
            fraud_score=fraud_score,
            is_anomaly=final_anomaly,
            model_version=MODEL_VERSION,
            processing_time_ms=processing_time
        )

    except Exception as e:
        state.error_count += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(batch: TransactionBatch):
    """
    Predict fraud for a batch of transactions.

    This is the recommended endpoint for high-throughput scenarios.
    Processes transactions in order to maintain proper feature state.
    """
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    predictions: List[PredictionResult] = []
    anomaly_count = 0

    try:
        # Process each transaction in order (required for streaming features)
        feature_vectors = []
        for tx in batch.transactions:
            tx_dict = tx.model_dump()
            feature_vector = state.feature_state.get_feature_vector(tx_dict)
            feature_vectors.append(feature_vector)

        # Batch prediction
        if feature_vectors:
            X = np.array(feature_vectors)
            fraud_scores = state.model.predict_proba(X)[:, 1]

            # Iterate through transactions and scores sequentially
            for tx, score in zip(batch.transactions, fraud_scores):
                tx_dict = tx.model_dump()
                fraud_score = float(score)
                base_anomaly = fraud_score >= state.config.anomaly_threshold
                user_id = tx_dict.get(state.config.sender_col)
                ts_raw = tx_dict.get(state.config.timestamp_col)
                country = None
                if hasattr(state.config, 'sender_country_col') and state.config.sender_country_col:
                    country = tx_dict.get(state.config.sender_country_col)
                # Parse timestamp
                try:
                    current_ts = pd.to_datetime(ts_raw) if ts_raw is not None else None
                except Exception:
                    current_ts = None
                # Geographic rule
                geo_anomaly = False
                if user_id is not None and current_ts is not None:
                    last_info = state.user_last_tx_info.get(user_id)
                    if last_info:
                        last_ts, last_country = last_info
                        if last_country and country and last_country != country:
                            if (current_ts - last_ts).total_seconds() <= state.geo_window_seconds:
                                geo_anomaly = True
                # Sequential rule
                seq_anomaly = False
                if user_id is not None:
                    seq_anomaly = state.user_consecutive_anomalies.get(user_id, 0) >= state.sequential_threshold
                final_anomaly = base_anomaly or geo_anomaly or seq_anomaly
                # Update counters
                if user_id is not None:
                    state.user_consecutive_anomalies[user_id] = (
                        state.user_consecutive_anomalies.get(user_id, 0) + 1
                    ) if final_anomaly else 0
                    state.user_last_tx_info[user_id] = (current_ts, country)
                if final_anomaly:
                    anomaly_count += 1
                predictions.append(PredictionResult(
                    transaction_id=tx.transaction_id,
                    fraud_score=fraud_score,
                    is_anomaly=final_anomaly,
                    model_version=MODEL_VERSION,
                    processing_time_ms=0  # calculated at batch level
                ))

        processing_time = (time.time() - start_time) * 1000
        throughput = len(batch.transactions) / (processing_time / 1000) if processing_time > 0 else 0

        state.prediction_count += len(batch.transactions)

        return BatchPredictionResult(
            predictions=predictions,
            total_transactions=len(batch.transactions),
            anomaly_count=anomaly_count,
            processing_time_ms=processing_time,
            throughput_tps=throughput
        )

    except Exception as e:
        state.error_count += 1
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    metrics = []

    # Counter metrics
    metrics.append(f'fraud_predictions_total {state.prediction_count}')
    metrics.append(f'fraud_errors_total {state.error_count}')

    # Gauge metrics
    metrics.append(f'fraud_model_loaded {1 if state.is_loaded else 0}')

    if state.load_time:
        uptime = (datetime.utcnow() - state.load_time).total_seconds()
        metrics.append(f'fraud_service_uptime_seconds {uptime}')

    return "\n".join(metrics)


@app.post("/reset-state")
async def reset_feature_state():
    """Reset the streaming feature state (for testing)."""
    if state.config:
        state.feature_state = StreamingFeatureState(state.config)
        state.user_consecutive_anomalies = defaultdict(int)
        state.user_last_tx_info = {}
        return {"status": "reset", "message": "Feature state and rule state cleared"}
    return {"status": "error", "message": "Config not loaded"}


@app.get("/config")
async def get_config():
    """Get current model configuration."""
    if not state.config:
        raise HTTPException(status_code=503, detail="Config not loaded")

    return {
        "anomaly_threshold": state.config.anomaly_threshold,
        "feature_count": len(state.config.feature_columns),
        "feature_columns": state.config.feature_columns,
        "velocity_windows": state.config.velocity_windows_minutes,
        "max_history_per_user": state.config.max_history_per_user
    }


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1
):
    """Start the FastAPI server."""
    uvicorn.run(
        "src.streaming.predict_service.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fraud Detection Predict Service')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--config', type=str, default=CONFIG_PATH)

    args = parser.parse_args()

    # Override paths if provided
    MODEL_PATH = args.model
    CONFIG_PATH = args.config

    uvicorn.run(app, host=args.host, port=args.port)