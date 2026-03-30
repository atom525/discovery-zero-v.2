# Compatibility shim: re-export from vendored local engine
from gaia_bp.engine import *  # noqa: F401,F403
from gaia_bp.engine import EngineConfig, InferenceEngine  # noqa: F811
