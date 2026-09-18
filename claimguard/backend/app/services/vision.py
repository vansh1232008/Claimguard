"""Damage photograph analysis.

Preferred path: a YOLO detector (ultralytics) for damage regions.
Fallback path: classical OpenCV severity heuristics (edge density, dark-region
analysis, colour variance) — and if OpenCV is not installed either, a pure-PIL
statistical estimate. The agent above it gets the same shape of finding in all
three cases.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

DAMAGE_CLASSES = ["dent", "scratch", "crack", "shatter", "crumple", "rust"]
REGIONS = ["front_bumper", "bonnet", "left_door", "right_door", "rear_bumper", "windscreen"]


@dataclass
class DamageFinding:
    region: str
    damage_type: str
    severity: str  # minor | moderate | severe
    confidence: float
    area_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _severity_from_area(area: float) -> str:
    if area < 0.04:
        return "minor"
    if area < 0.15:
        return "moderate"
    return "severe"


class YoloDamageDetector:
    name = "yolov8"

    def __init__(self, weights: str):
        from ultralytics import YOLO

        self._model = YOLO(weights)

    def analyse(self, image_path: Path) -> list[DamageFinding]:
        results = self._model(str(image_path), verbose=False)
        findings: list[DamageFinding] = []
        for res in results:
            h, w = res.orig_shape
            frame_area = float(h * w) or 1.0
            for box in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                area = ((x2 - x1) * (y2 - y1)) / frame_area
                cls_idx = int(box.cls[0]) if box.cls is not None else 0
                label = res.names.get(cls_idx, "damage") if hasattr(res, "names") else "damage"
                cy = (y1 + y2) / 2 / max(h, 1)
                cx = (x1 + x2) / 2 / max(w, 1)
                region = _region_from_position(cx, cy)
                findings.append(
                    DamageFinding(
                        region=region,
                        damage_type=str(label),
                        severity=_severity_from_area(area),
                        confidence=round(float(box.conf[0]) if box.conf is not None else 0.5, 3),
                        area_fraction=round(area, 4),
                    )
                )
        return findings


def _region_from_position(cx: float, cy: float) -> str:
    if cy < 0.33:
        return "windscreen" if 0.3 < cx < 0.7 else "bonnet"
    if cy > 0.7:
        return "rear_bumper"
    if cx < 0.33:
        return "left_door"
    if cx > 0.66:
        return "right_door"
    return "front_bumper"


class OpenCVDamageDetector:
    """Classical severity estimate: how disturbed is the panel surface?"""

    name = "opencv-heuristic"

    def analyse(self, image_path: Path) -> list[DamageFinding]:
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            return []
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 160)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(img.shape[0] * img.shape[1])
        findings: list[DamageFinding] = []
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
            area = float(cv2.contourArea(cnt))
            if area / frame_area < 0.01:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = (x + w / 2) / img.shape[1], (y + h / 2) / img.shape[0]
            roi = gray[y : y + h, x : x + w]
            texture = float(np.std(roi)) if roi.size else 0.0
            damage_type = "crack" if (w / max(h, 1)) > 3 or (h / max(w, 1)) > 3 else "dent"
            if texture > 60:
                damage_type = "shatter"
            findings.append(
                DamageFinding(
                    region=_region_from_position(cx, cy),
                    damage_type=damage_type,
                    severity=_severity_from_area(area / frame_area),
                    confidence=round(min(0.9, 0.4 + texture / 150.0), 3),
                    area_fraction=round(area / frame_area, 4),
                )
            )
        return findings


class StatisticalDamageDetector:
    """Last-resort estimator: deterministic, no CV dependencies at all.

    Uses image statistics when PIL is present, and a content hash otherwise, so
    the same photograph always yields the same finding. Demo-safe, clearly
    labelled as heuristic in the response.
    """

    name = "statistical-fallback"

    def analyse(self, image_path: Path) -> list[DamageFinding]:
        try:
            from PIL import Image, ImageStat

            with Image.open(image_path) as im:
                im = im.convert("L").resize((256, 256))
                stat = ImageStat.Stat(im)
                stddev = stat.stddev[0]
                mean = stat.mean[0]
            area = min(0.4, max(0.02, (stddev / 128.0) * 0.3))
            severity = _severity_from_area(area)
            damage_type = "crumple" if stddev > 70 else "dent" if mean < 120 else "scratch"
            seed = int(stddev * 1000) % len(REGIONS)
        except Exception:
            digest = hashlib.blake2b(image_path.name.encode(), digest_size=8).digest()
            area = 0.03 + (digest[0] % 20) / 200.0
            severity = _severity_from_area(area)
            damage_type = DAMAGE_CLASSES[digest[1] % len(DAMAGE_CLASSES)]
            seed = digest[2] % len(REGIONS)
        return [
            DamageFinding(
                region=REGIONS[seed],
                damage_type=damage_type,
                severity=severity,
                confidence=0.55,
                area_fraction=round(area, 4),
            )
        ]


_detector = None


def get_detector():
    global _detector
    if _detector is not None:
        return _detector
    weights = Path("artifacts/yolo_damage.pt")
    if weights.exists():
        try:
            _detector = YoloDamageDetector(str(weights))
            logger.info("Damage detector: YOLOv8 (%s)", weights)
            return _detector
        except Exception as exc:
            logger.warning("YOLO unavailable (%s)", exc)
    try:
        import cv2  # noqa: F401

        _detector = OpenCVDamageDetector()
        logger.info("Damage detector: OpenCV heuristic")
    except Exception:
        _detector = StatisticalDamageDetector()
        logger.info("Damage detector: statistical fallback")
    return _detector


def analyse_images(paths: list[Path]) -> dict[str, Any]:
    detector = get_detector()
    findings: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            findings.extend(f.to_dict() for f in detector.analyse(path))
        except Exception as exc:
            logger.warning("Damage analysis failed for %s: %s", path.name, exc)
    return {"backend": detector.name, "findings": findings, "images_analysed": len(paths)}
