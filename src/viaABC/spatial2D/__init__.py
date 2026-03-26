# src/viaABC/spatial2D/__init__.py
try:
    from ._grid_core import GridCore
except Exception:  
    GridCore = None

__all__ = ["GridCore"]