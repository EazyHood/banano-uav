"""Pruebas de visualizacion (que generen archivos sin error)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.pipeline import detect_banana
from banano.synth import synth_plantation
from banano.visualize import overlay, save_score_map


def test_overlay_and_score_map(tmp_path):
    img, _, _, _ = synth_plantation(H=400, W=400, gsd_cm=3.0, seed=2)
    res = detect_banana(img, gsd_cm=3.0)
    p1 = os.path.join(str(tmp_path), "ov.png")
    p2 = os.path.join(str(tmp_path), "sc.png")
    overlay(img, res, out_path=p1)
    save_score_map(res, p2)
    assert os.path.exists(p1) and os.path.getsize(p1) > 0
    assert os.path.exists(p2) and os.path.getsize(p2) > 0
