"""Pinned H1 appearance feature extractors."""

from .base import FeatureExtractor, ModelContractError, validate_embeddings
from .factory import REAL_MODEL_NAMES, create_extractor, model_signature
from .topic_agw import TopicCompatibilityError

__all__ = [
    "FeatureExtractor",
    "ModelContractError",
    "REAL_MODEL_NAMES",
    "TopicCompatibilityError",
    "create_extractor",
    "model_signature",
    "validate_embeddings",
]
