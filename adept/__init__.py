import os as _os

# Set by lightweight worker subprocesses (e.g. the OSIRIS stream-drain workers,
# `python -m adept.osiris.stream --drain-worker`) that need `adept.osiris.*`
# without paying for jax and the solver stack (~12 s import, hundreds of MB RSS
# per process on Perlmutter).
if not _os.environ.get("ADEPT_SKIP_SOLVER_IMPORTS"):
    from ._base_ import ADEPTModule, ergoExo  # noqa: I001
    from .mlflow_logging import MlflowLoggingModule
    from . import hermite_legendre_1d, hermite_poisson_1d, lpse2d, vlasov1d, vlasov2d
