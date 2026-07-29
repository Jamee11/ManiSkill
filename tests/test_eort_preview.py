import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/eort/render_objectcentric_preview.py"
SPEC = importlib.util.spec_from_file_location("eort_preview", SCRIPT)
preview = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preview
SPEC.loader.exec_module(preview)


class Labels:
    files = ("object_bbox_xyxy", "goal_bbox_xyxy", "object_visible", "goal_visible", "object_mask_centroid_uv", "goal_mask_centroid_uv", "physical_contact", "push_interaction_phase")

    def __init__(self):
        self.values = {
            "object_bbox_xyxy": np.array([[1, 2, 5, 6]]), "goal_bbox_xyxy": np.array([[6, 3, 10, 8]]),
            "object_visible": np.array([[True]]), "goal_visible": np.array([[True]]),
            "object_mask_centroid_uv": np.array([[3, 4]]), "goal_mask_centroid_uv": np.array([[8, 5]]),
            "physical_contact": np.array([[True]]), "push_interaction_phase": np.array([[2]]),
        }

    def __getitem__(self, key):
        return self.values[key]


def test_preview_annotation_draws_derived_labels():
    annotated = preview.annotate(np.zeros((16, 16, 3), dtype=np.uint8), Labels(), 0)
    assert annotated.shape == (16, 16, 3)
    assert annotated.any()
