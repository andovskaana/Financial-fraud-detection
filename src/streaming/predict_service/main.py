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

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
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

    # R12/R13: SHAP explainer (initialised after model load)
    explainer = None
    shap_feature_hits: Dict[str, int] = {}  # feature_name -> hit count


state = ModelState()


def compute_shap_vector(feature_vector: np.ndarray) -> Optional[np.ndarray]:
    """Return per-feature SHAP contributions for one transaction.

    For XGBoost models, avoid shap.TreeExplainer because newer XGBoost stores
    base_score as '[5E-1]', which some SHAP versions cannot parse. XGBoost's
    native pred_contribs=True is exact Tree SHAP and returns n_features + bias.
    """
    try:
        import xgboost as xgb
        if isinstance(state.model, xgb.XGBClassifier):
            feature_cols = state.config.feature_columns
            x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
            dmat = xgb.DMatrix(x, feature_names=feature_cols)
            contribs = state.model.get_booster().predict(
                dmat, pred_contribs=True, validate_features=False
            )
            if contribs.ndim == 3:
                if contribs.shape[1] == len(feature_cols) + 1:
                    contribs = contribs[:, :, 1]
                else:
                    contribs = contribs[:, 1, :]
            return contribs[0, :-1]  # drop bias column
    except Exception as exc:
        logger.debug(f"XGBoost native SHAP failed: {exc}")

    if state.explainer is None:
        return None

    sv = state.explainer.shap_values(np.asarray(feature_vector, dtype=np.float64).reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]
    if hasattr(sv, 'values'):
        sv = sv.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv[0]

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
    top_shap_features: Optional[List[Dict[str, Any]]] = None  # R13: top-3 SHAP features for fraud


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

        # R12/R13: initialise SHAP. For XGBoost, compute SHAP with native
        # Booster.predict(pred_contribs=True) at prediction time because newer
        # XGBoost base_score values such as '[5E-1]' break shap.TreeExplainer.
        try:
            import xgboost as xgb
            if isinstance(state.model, xgb.XGBClassifier):
                state.explainer = None
                state.shap_feature_hits = defaultdict(int)
                logger.info("SHAP enabled via XGBoost native pred_contribs=True")
            elif HAS_SHAP:
                state.explainer = shap.TreeExplainer(state.model)
                state.shap_feature_hits = defaultdict(int)
                logger.info("SHAP TreeExplainer initialised")
            else:
                logger.warning("shap package not installed — per-prediction SHAP disabled")
        except Exception as shap_err:
            logger.warning(f"Could not initialise SHAP support: {shap_err}")
            state.explainer = None

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

        # R5: update previous fraud count after this transaction is evaluated.
        state.feature_state.update_fraud_count(tx_dict, final_anomaly)

        # R12/R13: compute top-3 SHAP features for fraud predictions
        top_shap = None
        if final_anomaly:
            try:
                sv = compute_shap_vector(feature_vector)
                if sv is not None:
                    feature_cols = state.config.feature_columns
                    top3 = sorted(
                        zip(feature_cols, sv),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:3]
                    top_shap = [
                        {'feature': f, 'shap_value': round(float(v), 4)}
                        for f, v in top3
                    ]
                    # Increment per-feature hit counter (R13 Prometheus)
                    for f, _ in top3:
                        state.shap_feature_hits[f] = state.shap_feature_hits.get(f, 0) + 1
            except Exception as shap_err:
                logger.debug(f"SHAP per-prediction failed: {shap_err}")

        processing_time = (time.time() - start_time) * 1000
        state.prediction_count += 1

        return PredictionResult(
            transaction_id=transaction.transaction_id,
            fraud_score=fraud_score,
            is_anomaly=final_anomaly,
            model_version=MODEL_VERSION,
            processing_time_ms=processing_time,
            top_shap_features=top_shap
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
        # Process each transaction in order (required for streaming features).
        # We intentionally predict one-by-one here so prev_fraud_count for later
        # transactions in the same API batch can include earlier fraud/anomaly
        # results from the same user.
        for tx in batch.transactions:
            tx_dict = tx.model_dump()

            feature_vector = state.feature_state.get_feature_vector(tx_dict)
            fraud_score = float(state.model.predict_proba(
                feature_vector.reshape(1, -1)
            )[0, 1])

            base_anomaly = fraud_score >= state.config.anomaly_threshold
            user_id = tx_dict.get(state.config.sender_col)
            ts_raw = tx_dict.get(state.config.timestamp_col)
            country = None
            if hasattr(state.config, 'sender_country_col') and state.config.sender_country_col:
                country = tx_dict.get(state.config.sender_country_col)

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

            # R5: update previous fraud count after current evaluation.
            state.feature_state.update_fraud_count(tx_dict, final_anomaly)

            # R12/R13: compute top-3 SHAP features for fraud predictions
            top_shap = None
            if final_anomaly:
                try:
                    sv = compute_shap_vector(feature_vector)
                    if sv is not None:
                        feature_cols = state.config.feature_columns
                        top3 = sorted(
                            zip(feature_cols, sv),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:3]
                        top_shap = [
                            {'feature': f, 'shap_value': round(float(v), 4)}
                            for f, v in top3
                        ]
                        for f, _ in top3:
                            state.shap_feature_hits[f] = state.shap_feature_hits.get(f, 0) + 1
                except Exception as shap_err:
                    logger.debug(f"SHAP per-prediction (batch) failed: {shap_err}")

            if final_anomaly:
                anomaly_count += 1

            predictions.append(PredictionResult(
                transaction_id=tx.transaction_id,
                fraud_score=fraud_score,
                is_anomaly=final_anomaly,
                model_version=MODEL_VERSION,
                processing_time_ms=0,  # calculated at batch level
                top_shap_features=top_shap
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

    # R13: SHAP feature hit counters
    for feature, count in state.shap_feature_hits.items():
        safe_feat = feature.replace(' ', '_').replace('-', '_')
        metrics.append(f'shap_feature_hits_total{{feature="{safe_feat}"}} {count}')

    return "\n".join(metrics)


@app.get("/shap-stats")
async def get_shap_stats():
    """Return accumulated SHAP feature hit counts (R13)."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sorted_hits = sorted(
        state.shap_feature_hits.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return {
        "shap_available": state.explainer is not None,
        "total_fraud_predictions_with_shap": sum(state.shap_feature_hits.values()),
        "top_features": [
            {"feature": f, "hit_count": c}
            for f, c in sorted_hits[:20]
        ]
    }


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