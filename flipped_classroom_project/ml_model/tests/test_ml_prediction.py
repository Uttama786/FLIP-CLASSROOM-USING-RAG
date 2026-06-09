"""
Tests for ml_model/prediction.py
Covers: predict_student() with mock models, boundary conditions,
at-risk classification, and chart path helpers.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
from django.test import TestCase


# ──────────────────────────────────────────────────────────────
# Helper: build fake sklearn-compatible mock models
# ──────────────────────────────────────────────────────────────

def _build_mock_models(predicted_score=75.0, predicted_class=0, n_classes=4):
    """Create a tuple of (scaler, label_encoder, rf_regressor, rf_classifier) mocks."""
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 7))

    le = MagicMock()
    # Simulate 4 performance classes
    le.inverse_transform.return_value = ["High"]

    rf_reg = MagicMock()
    rf_reg.predict.return_value = np.array([predicted_score])

    rf_cls = MagicMock()
    rf_cls.predict.return_value = np.array([predicted_class])
    proba = np.zeros((1, n_classes))
    proba[0][predicted_class] = 0.9
    rf_cls.predict_proba.return_value = proba

    return scaler, le, rf_reg, rf_cls


def _sample_features(**overrides):
    base = {
        "videos_watched": 5,
        "total_video_time_minutes": 120.0,
        "quiz_avg_score": 7.5,
        "assignment_avg_marks": 15.0,
        "attendance_percentage": 80.0,
        "participation_score": 6.0,
        "previous_gpa": 8.0,
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────
# predict_student — basic contract
# ──────────────────────────────────────────────────────────────

class PredictStudentTest(TestCase):
    def setUp(self):
        from ml_model.prediction import predict_student
        self.predict = predict_student

    def test_returns_dict_with_required_keys(self):
        models = _build_mock_models(predicted_score=70.0)
        models[1].inverse_transform.return_value = ["High"]
        result = self.predict(_sample_features(), models_tuple=models)
        self.assertIn("predicted_score", result)
        self.assertIn("predicted_label", result)
        self.assertIn("is_at_risk", result)
        self.assertIn("confidence", result)

    def test_predicted_score_rounded_to_2dp(self):
        models = _build_mock_models(predicted_score=73.123456)
        models[1].inverse_transform.return_value = ["High"]
        result = self.predict(_sample_features(), models_tuple=models)
        # Should be rounded to 2 decimal places
        score_str = str(result["predicted_score"])
        decimal_part = score_str.split(".")[-1] if "." in score_str else ""
        self.assertLessEqual(len(decimal_part), 2)

    def test_confidence_is_percentage(self):
        models = _build_mock_models(predicted_score=80.0)
        models[1].inverse_transform.return_value = ["High"]
        result = self.predict(_sample_features(), models_tuple=models)
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 100)

    def test_score_clipped_at_100(self):
        models = _build_mock_models(predicted_score=120.0)
        models[1].inverse_transform.return_value = ["High"]
        result = self.predict(_sample_features(), models_tuple=models)
        self.assertLessEqual(result["predicted_score"], 100.0)

    def test_score_clipped_at_0(self):
        models = _build_mock_models(predicted_score=-10.0)
        models[1].inverse_transform.return_value = ["Low"]
        result = self.predict(_sample_features(), models_tuple=models)
        self.assertGreaterEqual(result["predicted_score"], 0.0)

    def test_missing_feature_defaults_to_zero(self):
        """Features not in the dict should default to 0."""
        models = _build_mock_models(predicted_score=50.0)
        models[1].inverse_transform.return_value = ["Medium"]
        # Only pass partial features
        partial = {"videos_watched": 3}
        result = self.predict(partial, models_tuple=models)
        self.assertIn("predicted_score", result)


# ──────────────────────────────────────────────────────────────
# At-Risk Classification Logic
# ──────────────────────────────────────────────────────────────

class AtRiskClassificationTest(TestCase):
    def setUp(self):
        from ml_model.prediction import predict_student
        self.predict = predict_student

    def _predict_with_label(self, score, label):
        models = _build_mock_models(predicted_score=score)
        models[1].inverse_transform.return_value = [label]
        return self.predict(_sample_features(), models_tuple=models)

    def test_high_performer_not_at_risk(self):
        result = self._predict_with_label(85.0, "High")
        self.assertFalse(result["is_at_risk"])
        self.assertEqual(result["predicted_label"], "High")

    def test_medium_performer_not_at_risk(self):
        result = self._predict_with_label(60.0, "Medium")
        self.assertFalse(result["is_at_risk"])

    def test_low_performer_is_at_risk(self):
        result = self._predict_with_label(42.0, "Low")
        self.assertTrue(result["is_at_risk"])

    def test_at_risk_label_is_at_risk(self):
        result = self._predict_with_label(30.0, "At-Risk")
        self.assertTrue(result["is_at_risk"])

    def test_score_below_40_always_at_risk(self):
        """Even if label is not At-Risk, score < 40 triggers at-risk."""
        result = self._predict_with_label(35.0, "Medium")
        self.assertTrue(result["is_at_risk"])

    def test_score_exactly_40_not_at_risk_if_medium(self):
        result = self._predict_with_label(40.0, "Medium")
        self.assertFalse(result["is_at_risk"])


# ──────────────────────────────────────────────────────────────
# predict_student — no models on disk
# ──────────────────────────────────────────────────────────────

class PredictStudentNoModelsTest(TestCase):
    @patch("ml_model.prediction._load_models", side_effect=FileNotFoundError)
    def test_raises_runtime_error_when_models_missing(self, _mock):
        from ml_model.prediction import predict_student
        with self.assertRaises(RuntimeError) as ctx:
            predict_student(_sample_features())
        self.assertIn("model", str(ctx.exception).lower())


# ──────────────────────────────────────────────────────────────
# Feature importance / model comparison chart paths
# ──────────────────────────────────────────────────────────────

class ChartPathTests(TestCase):
    def test_feature_importance_chart_none_when_missing(self):
        from ml_model.prediction import get_feature_importance_chart
        with patch("os.path.exists", return_value=False):
            result = get_feature_importance_chart()
        self.assertIsNone(result)

    def test_feature_importance_chart_path_when_exists(self):
        from ml_model.prediction import get_feature_importance_chart
        with tempfile.TemporaryDirectory() as tmp:
            with patch("ml_model.prediction.BASE_DIR", tmp):
                plots_dir = os.path.join(tmp, "plots")
                os.makedirs(plots_dir)
                chart_path = os.path.join(plots_dir, "rf_classification_feature_importance.png")
                with open(chart_path, "w") as f:
                    f.write("fake png")
                result = get_feature_importance_chart()
        self.assertIsNotNone(result)

    def test_model_comparison_chart_none_when_missing(self):
        from ml_model.prediction import get_model_comparison_chart
        with patch("os.path.exists", return_value=False):
            result = get_model_comparison_chart()
        self.assertIsNone(result)

    def test_model_comparison_chart_path_when_exists(self):
        from ml_model.prediction import get_model_comparison_chart
        with tempfile.TemporaryDirectory() as tmp:
            with patch("ml_model.prediction.BASE_DIR", tmp):
                plots_dir = os.path.join(tmp, "plots")
                os.makedirs(plots_dir)
                chart_path = os.path.join(plots_dir, "model_comparison.png")
                with open(chart_path, "w") as f:
                    f.write("fake png")
                result = get_model_comparison_chart()
        self.assertIsNotNone(result)


# ──────────────────────────────────────────────────────────────
# FEATURES list contract
# ──────────────────────────────────────────────────────────────

class FeaturesListTest(TestCase):
    def test_features_list_has_expected_fields(self):
        from ml_model.prediction import FEATURES
        expected = {
            "videos_watched",
            "total_video_time_minutes",
            "quiz_avg_score",
            "assignment_avg_marks",
            "attendance_percentage",
            "participation_score",
            "previous_gpa",
        }
        self.assertEqual(set(FEATURES), expected)

    def test_features_list_has_7_items(self):
        from ml_model.prediction import FEATURES
        self.assertEqual(len(FEATURES), 7)


# ──────────────────────────────────────────────────────────────
# _load_models helper
# ──────────────────────────────────────────────────────────────

class LoadModelsTest(TestCase):
    @patch("joblib.load")
    def test_load_models_calls_joblib_four_times(self, mock_load):
        mock_load.return_value = MagicMock()
        from ml_model.prediction import _load_models
        scaler, le, rf_reg, rf_cls = _load_models()
        self.assertEqual(mock_load.call_count, 4)

    @patch("joblib.load", side_effect=FileNotFoundError("model not found"))
    def test_load_models_propagates_file_not_found(self, _mock):
        from ml_model.prediction import _load_models
        with self.assertRaises(FileNotFoundError):
            _load_models()
