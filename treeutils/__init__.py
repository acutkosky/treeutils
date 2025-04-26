"""Tree utilities for Python objects."""
from treeutils.core import *
from treeutils.default_nodes import *
try:
    import torch
    from treeutils.pytorch_nodes import *
except ImportError:
    pass
