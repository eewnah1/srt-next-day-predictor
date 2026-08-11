"""Shim that loads the shared multi-horizon engine using this repo's config.py."""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if "config" in sys.modules:
    del sys.modules["config"]

_engine_name = f"multi_horizon_engine_{os.path.basename(os.path.dirname(__file__))}"
_spec = importlib.util.spec_from_file_location(
    _engine_name,
    "/home/ubuntu/repos/hst-next-day-predictor/multi_horizon.py",
)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)

sys.modules[__name__] = _engine

if __name__ == "__main__":
    import json
    res = _engine.run_pipeline(force_retrain=True)
    print(json.dumps(res, indent=2, default=str))
