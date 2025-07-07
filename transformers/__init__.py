"""
Transformers Core Utilities

This module contains the essential transformers infrastructure needed for GLM-4V.
"""

# Version information
from .__version__ import __version__

# Essential utilities only to avoid circular imports
try:
    from .configuration_utils import PretrainedConfig
except ImportError:
    PretrainedConfig = None

try:
    from .modeling_utils import PreTrainedModel
except ImportError:
    PreTrainedModel = None

try:
    from .modeling_outputs import BaseModelOutputWithPast, ModelOutput
except ImportError:
    BaseModelOutputWithPast = None
    ModelOutput = None

# Core infrastructure
__all__ = [
    "__version__",
    "PretrainedConfig",
    "PreTrainedModel", 
    "BaseModelOutputWithPast",
    "ModelOutput",
] 