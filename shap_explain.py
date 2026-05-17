"""
shap_explain.py — Standalone SHAP Explainability (R12 + R13)
=============================================================
Run AFTER training is complete. Loads the saved model, config, and
test dataset, then produces:

  models/shap_importance.json        — ranked mean-|SHAP| per feature
  models/plots/shap_summary_bar.png  — global bar chart
  models/plots/shap_beeswarm.png     — beeswarm dot plot

Usage
-----
  python shap_explain.py
  python shap_explain.py --data models/data/financial_fraud_detection_dataset.csv
                         --model models/fraud_model.joblib
                         --config models/feature_config.json
                         --output models
                         --samples 500   # number of test rows to explain
                         --debug         # print dtype info before SHAP runs
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.training.features import FeatureConfig, ColumnDetector, BatchFeatureEngineer


# ── helpers ─────────────────────────────────────────────────────────────────

def _force_float64(df: pd.DataFrame, feature_cols: list, debug: bool = False) -> np.ndarray:
    """
    Convert a DataFrame subset to a clean float64 numpy array.

    Strategy (applied column by column so bad columns are visible):
      1. pd.to_numeric(errors='coerce')  — handles '[5E-1]', 'unknown', etc.
      2. fillna(0)                        — NaN from coerce or real missing
      3. astype(np.float64)              — final cast; will never fail after step 1-2
    """
    out = np.zeros((len(df), len(feature_cols)), dtype=np.float64)

    for i, col in enumerate(feature_cols):
        series = df[col]
        if debug and series.dtype == object:
            bad = series[pd.to_numeric(series, errors='coerce').isna() & series.notna()]
            if len(bad):
                print(f"  [debug] '{col}' has {len(bad)} non-numeric values, "
                      f"e.g. {bad.iloc[0]!r}  → coerced to 0")
        out[:, i] = pd.to_numeric(series, errors='coerce').fillna(0).to_numpy(dtype=np.float64)

    return out


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Standalone SHAP explainability')
    parser.add_argument('--data',    default='models/data/financial_fraud_detection_dataset.csv')
    parser.add_argument('--model',   default='models/fraud_model.joblib')
    parser.add_argument('--config',  default='models/feature_config.json')
    parser.add_argument('--output',  default='models')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of test-set rows to compute SHAP on')
    parser.add_argument('--debug',   action='store_true',
                        help='Print dtype/value info for every problem column')
    args = parser.parse_args()

    try:
        import shap
    except ImportError:
        print('[ERROR] shap is not installed. Run: pip install shap>=0.44.0')
        sys.exit(1)

    output_path = Path(args.output)
    plots_dir   = output_path / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f'\nLoading model from {args.model} ...')
    model = joblib.load(args.model)
    print(f'  Model type: {type(model).__name__}')

    # ── 2. Load config ───────────────────────────────────────────────────────
    print(f'Loading config from {args.config} ...')
    config = FeatureConfig.load(args.config)
    feature_cols = config.feature_columns
    print(f'  Features: {len(feature_cols)}')

    # ── 3. Load & engineer a test slice ─────────────────────────────────────
    print(f'\nLoading dataset from {args.data} ...')
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    print(f'  Shape: {df.shape}')

    # Reproduce the same time-based 60/20/20 split used in training
    if config.timestamp_col and config.timestamp_col in df.columns:
        df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors='coerce')
        df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    n = len(df)
    test_start = int(n * 0.60)
    test_end   = int(n * 0.80)
    df_test = df.iloc[test_start:test_end].copy()
    print(f'  Test slice: {len(df_test):,} rows')

    # ── 4. Feature engineering (transform only — no fitting) ─────────────────
    print('\nRunning feature engineering on test slice ...')
    fe = BatchFeatureEngineer(config)
    df_test_fe = fe.transform(df_test)

    # ── 5. Build clean float64 array ─────────────────────────────────────────
    print('\nBuilding feature matrix (strict float64) ...')
    if args.debug:
        print('  Checking for non-numeric values column by column:')
    X_test = _force_float64(df_test_fe, feature_cols, debug=args.debug)
    print(f'  X_test dtype : {X_test.dtype}')
    print(f'  X_test shape : {X_test.shape}')
    print(f'  NaN count    : {np.isnan(X_test).sum()}')
    print(f'  Inf count    : {np.isinf(X_test).sum()}')

    # Replace any stray inf/nan (safety net)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 6. Sample rows for SHAP ──────────────────────────────────────────────
    n_shap = min(args.samples, len(X_test))
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(X_test), size=n_shap, replace=False)
    X_shap = X_test[idx]
    print(f'\nSHAP sample : {n_shap} rows')

    # ── 7/8. Compute SHAP values ─────────────────────────────────────────────
    # IMPORTANT:
    # SHAP TreeExplainer currently breaks with newer XGBoost models because
    # XGBoost stores base_score as a JSON list string, for example '[5E-1]'.
    # Instead of trying to patch the saved model JSON, use XGBoost's native
    # exact Tree SHAP implementation: Booster.predict(..., pred_contribs=True).
    # It returns one contribution per feature plus a final bias column.
    print('Computing SHAP values ...')
    import xgboost as xgb

    if isinstance(model, xgb.XGBClassifier):
        booster = model.get_booster()
        dmat = xgb.DMatrix(X_shap, feature_names=feature_cols)
        contribs = booster.predict(dmat, pred_contribs=True, validate_features=False)

        # Binary XGBoost usually returns: (n_samples, n_features + 1)
        # Multiclass can return 3-D; keep positive/fraud class when present.
        if contribs.ndim == 3:
            # Common shapes are (n_samples, n_classes, n_features + 1)
            # or (n_samples, n_features + 1, n_classes).
            if contribs.shape[1] == len(feature_cols) + 1:
                contribs = contribs[:, :, 1]
            else:
                contribs = contribs[:, 1, :]

        shap_values = contribs[:, :-1]  # remove expected-value / bias column
        print('  Used XGBoost native pred_contribs=True')
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    print(f'  shap_values shape : {shap_values.shape}')

    # ── 9. Global bar chart ──────────────────────────────────────────────────
    bar_path = plots_dir / 'shap_summary_bar.png'
    print(f'\nSaving bar chart → {bar_path}')
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_cols,
        plot_type='bar',
        show=False
    )
    plt.title('SHAP Feature Importance (mean |SHAP value|)')
    plt.tight_layout()
    plt.savefig(str(bar_path), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 10. Beeswarm plot ────────────────────────────────────────────────────
    bee_path = plots_dir / 'shap_beeswarm.png'
    print(f'Saving beeswarm   → {bee_path}')
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_cols,
        show=False
    )
    plt.title('SHAP Feature Impact (beeswarm)')
    plt.tight_layout()
    plt.savefig(str(bee_path), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 11. Save JSON ranking ────────────────────────────────────────────────
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_cols
    ).sort_values(ascending=False)

    json_path = output_path / 'shap_importance.json'
    mean_shap.to_json(str(json_path))
    print(f'Saved JSON        → {json_path}')

    # ── 12. Update model_metadata.json ──────────────────────────────────────
    meta_path = output_path / 'model_metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta['shap_top10_features'] = mean_shap.head(10).to_dict()
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'Updated metadata  → {meta_path}')

    # ── 13. Console summary ──────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(' Top 15 SHAP features')
    print(f'{"="*60}')
    for feat, val in mean_shap.head(15).items():
        print(f'  {feat:<42} {val:.5f}')

    print(f'\n✓ SHAP explainability complete.')
    print(f'  {bar_path}')
    print(f'  {bee_path}')
    print(f'  {json_path}')


if __name__ == '__main__':
    main()