from typing import List, Tuple, Dict, Any
from copy import copy

import torch

from treeutils.core import register_pytree_node


BUFFER = "buffer"
SUBMODULE = "submodule"
PARAMETER = "parameter"


def shallow_copy_module(module: torch.nn.Module) -> torch.nn.Module:
    """Shallow copy a PyTorch module.

    This function creates a level-one shallow copy of a PyTorch module.
    The lists of parameters, buffers and submodules are replaced with shallow copies.
    So, if you replace a parameter in the copy, it will not affect the original.
    Notice that this does NOT make copies of the underlying data in the parameters and buffers.
    """
    aux = copy(module)
    aux._modules = copy(module._modules)
    aux._parameters = copy(module._parameters)
    aux._buffers = copy(module._buffers)
    return aux


def flatten_module(module: torch.nn.Module) -> Tuple[List[Any], Any, List[Any]]:
    """Flatten a PyTorch module for pytree node registration.

    This function flattens only one level of the module.
    So, children will be a list of:
        * parameters and buffers accessible directly from this module (not submodules).
        * submodules themselves.

    aux will be tuple containing:
        * the module itself, with all children replaced by None.
        * a dictionary mapping child names to one of 'buffer', 'submodule', 'parameter'

    Args:
        module: The module to flatten

    Returns:
        A tuple of (children, aux, keys)
    """
    # Get all direct parameters and buffers
    children = []
    keys = []

    child_types = {}

    # Add parameters
    for name, param in module._parameters.items():
        children.append(param)
        keys.append(name)
        child_types[name] = PARAMETER
    # Add buffers
    for name, buffer in module._buffers.items():
        children.append(buffer)
        keys.append(name)
        child_types[name] = BUFFER
    # Add submodules
    for name, submodule in module._modules.items():
        children.append(submodule)
        keys.append(name)
        child_types[name] = SUBMODULE
    # Create a copy of the module with all children set to None
    blank_module = shallow_copy_module(module)
    aux = (blank_module, child_types)

    return children, aux, keys


def unflatten_module(
    children: List[Any], aux: Tuple[torch.nn.Module, Dict[str, str]], keys: List[Any]
) -> torch.nn.Module:
    """Unflatten a PyTorch module from its flattened components.

    Args:
        children: List of parameters, buffers, and submodules
        aux: Tuple containing:
            * The module with all children set to None
            * A dictionary mapping child names to one of 'buffer', 'submodule', 'parameter'
        keys: List of names for the children

    Returns:
        The reconstructed module
    """
    module, child_types = aux
    for key, child in zip(keys, children):
        if child_types[key] == PARAMETER:
            module._parameters[key] = child
        elif child_types[key] == SUBMODULE:
            module._modules[key] = child
        elif child_types[key] == BUFFER:
            module._buffers[key] = child
        else:
            raise ValueError(f"Unknown child type: {child_types[key]}")

    return module


# Register PyTorch Module as a pytree node
register_pytree_node(torch.nn.Module, flatten_module, unflatten_module)
