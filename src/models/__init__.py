"""
Models package initialization.
"""

from .evaluate import ErrorAnalysisResult, evaluate_champion_model
from .pipeline_factory import get_candidate_models_and_grids, get_model_pipeline
from .train import BenchmarkResult, CVFoldMetrics, ModelBenchmarkEngine

__all__ = [
    "get_model_pipeline",
    "get_candidate_models_and_grids",
    "ModelBenchmarkEngine",
    "BenchmarkResult",
    "CVFoldMetrics",
    "evaluate_champion_model",
    "ErrorAnalysisResult",
]
