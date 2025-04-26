"""Default node registrations for common Python types."""

from typing import List, Tuple, Dict, Any
from treeutils.core import register_pytree_node


def flatten_list(node: List[Any]) -> Tuple[List[Any], Any, List[Any]]:
    """Flatten a list node.

    Args:
        node: List to flatten

    Returns:
        A tuple of (children, None, None)
    """
    return node, None, list(range(len(node)))


def unflatten_list(children: List[Any], aux: Any, keys: List[Any]) -> List[Any]:
    """Unflatten a list node.

    Args:
        children: List of child nodes
        aux: Unused
        keys: Unused

    Returns:
        The reconstructed list
    """
    return [children[i] for i in keys]


def flatten_dict(node: Dict[Any, Any]) -> Tuple[List[Any], Any, List[Any]]:
    """Flatten a dict node.

    Args:
        node: Dict to flatten

    Returns:
        A tuple of (values, None, keys)
    """
    return list(node.values()), None, list(node.keys())


def unflatten_dict(children: List[Any], aux: Any, keys: List[Any]) -> Dict[Any, Any]:
    """Unflatten a dict node.

    Args:
        children: List of values
        aux: Unused
        keys: List of keys

    Returns:
        The reconstructed dict
    """
    return dict(zip(keys, children))


def flatten_tuple(node: Tuple[Any, ...]) -> Tuple[List[Any], Any, List[Any]]:
    """Flatten a tuple node.

    If the tuple has a _fields attribute, it is assumed to be a namedtuple and the keys will be the field names.
    Otherwise, the keys will be None.

    Args:
        node: Tuple to flatten

    Returns:
        A tuple of (elements, None, keys)
    """
    if hasattr(node, "_fields"):
        keys = list(node._fields)
    else:
        keys = list(range(len(node)))
    return list(node), None, keys


def unflatten_tuple(children: List[Any], aux: Any, keys: List[Any]) -> Tuple[Any, ...]:
    """Unflatten a tuple node.

    Args:
        children: List of elements
        aux: Unused
        keys: Unused

    Returns:
        The reconstructed tuple
    """
    return tuple(children)


# Register default nodes
register_pytree_node(list, flatten_list, unflatten_list)
register_pytree_node(dict, flatten_dict, unflatten_dict)
register_pytree_node(tuple, flatten_tuple, unflatten_tuple)
