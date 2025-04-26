from typing import List, Tuple, Dict, Any
from copy import copy

import torch

from treeutils.core import register_pytree_node


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

    aux will be the module itself, with all children replaced by None.
    keys will be a list of the names of the parameters and buffers and submodules needed to reconstruct the module.

    Args:
        module: The module to flatten

    Returns:
        A tuple of (children, aux, keys)
    """
    # Get all direct parameters and buffers
    children = []
    keys = []

    # Add parameters
    for name, param in module._parameters.items():
        children.append(param)
        keys.append(name)

    # Add buffers
    for name, buffer in module._buffers.items():
        children.append(buffer)
        keys.append(name)

    # Add submodules
    for name, submodule in module._modules.items():
        children.append(submodule)
        keys.append(name)

    # Create a copy of the module with all children set to None
    aux = shallow_copy_module(module)

    return children, aux, keys


def unflatten_module(
    children: List[Any], aux: torch.nn.Module, keys: List[Any]
) -> torch.nn.Module:
    """Unflatten a PyTorch module from its flattened components.

    Args:
        children: List of parameters, buffers, and submodules
        aux: The module with all children set to None
        keys: List of names for the children

    Returns:
        The reconstructed module
    """
    for key, child in zip(keys, children):
        if isinstance(getattr(aux, key), torch.nn.Parameter):
            aux._parameters[key] = child
        elif isinstance(getattr(aux, key), torch.nn.Module):
            aux._modules[key] = child
        else:
            aux._buffers[key] = child

    return aux


# Register PyTorch Module as a pytree node
register_pytree_node(torch.nn.Module, flatten_module, unflatten_module)
