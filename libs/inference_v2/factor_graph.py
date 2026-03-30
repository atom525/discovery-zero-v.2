# Compatibility shim: re-export from vendored local engine
from gaia_bp.factor_graph import *  # noqa: F401,F403
from gaia_bp.factor_graph import FactorType  # noqa: F811

OperatorType = FactorType
