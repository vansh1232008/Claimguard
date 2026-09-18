"""Fraud scoring service.

Loads the trained XGBoost model and the GraphSAGE claim embeddings from
``artifacts/`` when they exist. When they do not (fresh clone, model not
trained yet) it falls back to a transparent logistic scorecard so the platform
still produces calibrated-looking, explainable scores on day one.

Both paths return the same thing: a probability plus per-feature
contributions, which the anomaly agent turns into a written explanation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.core.logging import get_logger
from app.services.features import FEATURE_NAMES, build_feature_row, to_vector

logger = get_logger(__name__)


# Hand-tuned weights for the cold-start scorecard. Positive weight = more
# suspicious. Values are applied to z-scored features (see _SCALE below).
_SCORECARD_WEIGHTS: dict[str, float] = {
    "amount_to_limit_ratio": 1.35,
    "amount_vs_estimate_ratio": 1.10,
    "amount_to_premium_ratio": 0.55,
    "reporting_delay_days": 0.75,
    "days_since_policy_start": -0.80,
    "witnesses": -0.70,
    "police_report": -0.95,
    "injury_claimed": 0.45,
    "prior_claims_12m": 0.85,
    "customer_tenure_months": -0.60,
    "description_is_generic": 0.65,
    "missing_invoice": 0.80,
    "document_count": -0.35,
    "vendor_watchlisted": 1.25,
    "vendor_invoice_ratio": 0.70,
    "incident_is_weekend": 0.25,
    # Identity links are what actually distinguish a ring; raw component size is
    # inflated by incidental links (same garage, same postcode) so it is weak.
    "ring_size": 0.35,
    "ring_cohesion": 0.45,
    "shared_phone_degree": 1.05,
    "shared_account_degree": 1.20,
    "vendor_degree": 0.20,
}

# Rough centre/scale per feature so the scorecard behaves on raw values.
_SCALE: dict[str, tuple[float, float]] = {
    "amount_to_limit_ratio": (0.25, 0.25),
    "amount_vs_estimate_ratio": (1.0, 0.35),
    "amount_to_premium_ratio": (6.0, 6.0),
    "reporting_delay_days": (3.0, 7.0),
    "days_since_policy_start": (400.0, 300.0),
    "witnesses": (1.0, 1.0),
    "prior_claims_12m": (0.4, 1.0),
    "customer_tenure_months": (36.0, 30.0),
    "document_count": (2.0, 1.5),
    "vendor_claims_serviced": (25.0, 30.0),
    "vendor_invoice_ratio": (1.0, 0.5),
    "ring_size": (4.0, 6.0),
    "ring_cohesion": (0.2, 0.3),
    "shared_phone_degree": (0.2, 1.0),
    "shared_account_degree": (0.2, 1.0),
    "vendor_degree": (2.0, 4.0),
    "description_is_generic": (0.15, 0.4),
    "missing_invoice": (0.3, 0.5),
    "police_report": (0.5, 0.5),
    "injury_claimed": (0.12, 0.35),
    "incident_is_weekend": (0.3, 0.5),
    "amount_to_limit_ratio": (0.25, 0.25),
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class ScorecardModel:
    """Explainable logistic fallback. No training required."""

    name = "logistic-scorecard"
    trained = False

    def predict(self, row: dict[str, float]) -> tuple[float, list[dict[str, Any]]]:
        z = -2.60  # base-rate prior: ~7% at the population average feature vector
        contributions: list[dict[str, Any]] = []
        for feature, weight in _SCORECARD_WEIGHTS.items():
            raw = float(row.get(feature, 0.0))
            centre, scale = _SCALE.get(feature, (0.0, 1.0))
            norm = (raw - centre) / (scale or 1.0)
            norm = max(-4.0, min(4.0, norm))
            contrib = weight * norm
            z += contrib
            contributions.append({"name": feature, "value": round(raw, 4), "contribution": round(contrib, 4)})
        contributions.sort(key=lambda d: -abs(d["contribution"]))
        return _sigmoid(z), contributions


class XGBoostModel:
    name = "xgboost"
    trained = True

    def __init__(
        self,
        model_path: Path,
        feature_names: list[str],
        calibration: dict[str, Any] | None = None,
    ):
        import xgboost as xgb

        self._xgb = xgb
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))
        self.feature_names = feature_names
        # Platt parameters fitted on the validation fold at training time. The
        # model is trained with scale_pos_weight, so its raw output ranks well
        # but is not a probability - and the thresholds, the blend and the
        # number shown to the investigator all assume a probability.
        self.calibration = calibration or {}

    def _calibrate(self, prob: float) -> float:
        if self.calibration.get("method") != "platt":
            return prob
        p = min(max(prob, 1e-6), 1 - 1e-6)
        logit = math.log(p / (1 - p))
        return _sigmoid(self.calibration["a"] * logit + self.calibration["b"])

    def predict(self, row: dict[str, float]) -> tuple[float, list[dict[str, Any]]]:
        vec = np.asarray([to_vector(row, self.feature_names)], dtype=np.float32)
        dmat = self._xgb.DMatrix(vec, feature_names=self.feature_names)
        prob = self._calibrate(float(self.booster.predict(dmat)[0]))
        try:
            shap = self.booster.predict(dmat, pred_contribs=True)[0]
            contributions = [
                {
                    "name": self.feature_names[i],
                    "value": round(float(vec[0][i]), 4),
                    "contribution": round(float(shap[i]), 4),
                }
                for i in range(len(self.feature_names))
            ]
            contributions.sort(key=lambda d: -abs(d["contribution"]))
        except Exception:  # pragma: no cover
            contributions = []
        return prob, contributions


class FraudScorer:
    """Facade over whichever model is available, plus GraphSAGE embeddings."""

    def __init__(self) -> None:
        self.feature_names = list(FEATURE_NAMES)
        self.calibration: dict[str, Any] = {}
        self.model: Any = ScorecardModel()
        self.embeddings: dict[str, np.ndarray] = {}
        self._load()

    # -- loading ----------------------------------------------------------
    def _load(self) -> None:
        art = settings.artifacts
        meta_path = art / settings.feature_metadata_file
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                self.feature_names = meta.get("feature_names", self.feature_names)
                self.calibration = meta.get("calibration", {})
            except Exception as exc:
                logger.warning("Could not read feature metadata: %s", exc)

        model_path = art / settings.xgboost_model_file
        if model_path.exists():
            try:
                self.model = XGBoostModel(model_path, self.feature_names, self.calibration)
                logger.info(
                    "Fraud model: trained XGBoost (%s), calibration=%s",
                    model_path.name,
                    self.calibration.get("method", "none"),
                )
            except Exception as exc:
                logger.warning("Failed to load XGBoost model (%s); using scorecard", exc)
        else:
            logger.info(
                "Fraud model: logistic scorecard (no trained model in %s - run "
                "`python ml/train_xgboost.py` to upgrade)",
                art,
            )

        emb_path = art / settings.graphsage_embedding_file
        if emb_path.exists():
            try:
                blob = np.load(emb_path, allow_pickle=True)
                ids = list(blob["claim_ids"])
                vecs = blob["embeddings"]
                self.embeddings = {str(ids[i]): vecs[i] for i in range(len(ids))}
                logger.info("Loaded %d GraphSAGE embeddings", len(self.embeddings))
            except Exception as exc:
                logger.warning("Could not load GraphSAGE embeddings: %s", exc)

    def reload(self) -> None:
        self.__init__()  # type: ignore[misc]

    # -- scoring ----------------------------------------------------------
    def score(self, ctx: dict[str, Any]) -> dict[str, Any]:
        row = build_feature_row(ctx)
        prob, contributions = self.model.predict(row)
        return {
            "fraud_probability": round(float(prob), 4),
            "model": self.model.name,
            "trained": bool(getattr(self.model, "trained", False)),
            "top_features": contributions[:8],
            "features": row,
        }

    @property
    def mode(self) -> str:
        return self.model.name


_scorer: FraudScorer | None = None


def get_scorer() -> FraudScorer:
    global _scorer
    if _scorer is None:
        _scorer = FraudScorer()
    return _scorer
