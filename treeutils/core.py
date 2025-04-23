from typing import Callable, Type, Any, List, Tuple, NamedTuple, Optional, Dict

# A registry of pytree nodes
# pytree_node_registry[type] = (flatten_fn, unflatten_fn)
pytree_node_registry = {}


class RegistryEntry(NamedTuple):
    flatten_fn: Callable[[Any], Tuple[List[Any], Any, Optional[List[Any]]]]
    unflatten_fn: Callable[[Any, List[Any], Optional[List[Any]]], Any]


class KeyTuple(NamedTuple):
    type: Type
    key: Optional[Any]

    def __repr__(self) -> str:
        return f"{self.key}"


def get_type_name(type_obj: Type) -> str:
    """Get the name of a type as a string.

    Args:
        type_obj: The type object to get the name of

    Returns:
        The name of the type as a string
    """
    return type_obj.__name__


class PyTreeDef:
    """A tree structure definition for Python objects.
    
    This class represents the structure of a Python object tree, including its type,
    children, auxiliary data, and optional keys for named children.
    """
    def __init__(
        self,
        root: Any,
        children: List["PyTreeDef"],
        aux: Any,
        keys: Optional[List[Any]],
    ):
        """Initialize a PyTreeDef.
        
        Args:
            root: The root type of the tree (None if the node is a leaf)
            children: List of child PyTreeDefs
            aux: Auxiliary data associated with the node
            keys: Optional list of keys for named children
        """
        self.root = root
        self.children = children
        self.aux = aux
        self.keys = keys

    def as_string(self) -> str:
        """Convert the tree definition to a string representation.
        
        Returns:
            A string representation of the tree structure
        """
        if self.root is None:
            return "*"
        if self.keys is not None:
            children_string = [
                f"{key}: {child.as_string()}"
                for key, child in zip(self.keys, self.children)
            ]
        else:
            children_string = [child.as_string() for child in self.children]
        return f"{get_type_name(self.root)}[{', '.join(children_string)}]"

    def __repr__(self) -> str:
        """Get the string representation of the PyTreeDef.
        
        Returns:
            A string representation of the PyTreeDef
        """
        return f"PyTreeDef({self.as_string()})"


def register_pytree_node(
    cls: Type,
    flatten_fn: Callable[[Any], Tuple[List[Any], Any, Optional[List[Any]]]],
    unflatten_fn: Callable[[Any, List[Any]], Any],
) -> Type:
    """Register a new type as a pytree node.

    flatten_fn should take a single argument (the node to flatten) and return a tuple of (children, aux_data, keys).
    children is a list of the children of the node.
    aux_data is auxiliary data associated with the node used to reconstruct the node.
    keys is an optional list of keys for named children. It may also be needed to reconstruct the node.
    unflatten_fn should take three arguments (children, aux_data, keys) and return the reconstructed node.
    
    Args:
        cls: The type to register
        flatten_fn: Function to flatten instances of this type
        unflatten_fn: Function to unflatten instances of this type
        
    Returns:
        The registered type
        
    Raises:
        AssertionError: If the type is already registered
    """
    assert cls not in pytree_node_registry, f"PyTree node {cls} already registered!"
    pytree_node_registry[cls] = RegistryEntry(flatten_fn, unflatten_fn)
    return cls


def is_pytree_node(cls: Type) -> bool:
    """Check if a type is registered as a pytree node.
    
    Args:
        cls: The type to check
        
    Returns:
        True if the type is registered as a pytree node, False otherwise
    """
    for t in cls.__mro__:
        if t in pytree_node_registry:
            return True
    return False


def get_pytree_node_registry(cls: Type) -> RegistryEntry:
    """Get the registry entry for a pytree node type.
    
    Args:
        cls: The type to look up
        
    Returns:
        The registry entry for the type, or False if not found
    """
    for t in cls.__mro__:
        if t in pytree_node_registry:
            return pytree_node_registry[t]
    return False


def flatten_once(node: Any) -> Tuple[List[Any], Any, Optional[List[Any]]]:
    """Flatten a single pytree node.
    
    Args:
        node: The node to flatten
        
    Returns:
        A tuple of (children, aux_data, keys)
    """
    return get_pytree_node_registry(type(node)).flatten_fn(node)


def unflatten_once(
    node_type: Any, children: List[Any], aux: Any, keys: Optional[List[Any]]
) -> Any:
    """Unflatten a single pytree node.
    
    Args:
        node_type: The type of the node to unflatten
        children: List of child nodes
        aux: Auxiliary data
        keys: Optional list of keys for named children
        
    Returns:
        The reconstructed node
    """
    return get_pytree_node_registry(node_type).unflatten_fn(children, aux, keys)


def map(
    fn: Callable[[Any], Any],
    node: Any,
    with_path: bool = False,
    path_prefix: List[Any] = [],
    is_leaf: Callable[[Any], bool] = lambda *_: False,
) -> Any:
    """Map a function over a pytree structure.
    
    This function applies the given function to each node in the pytree, recursively traversing
    the tree structure. The function can optionally receive the path to each node.
    
    Args:
        fn: Function to apply to each node. If with_path is True, the function should accept
            two arguments (node, path). Otherwise, it should accept one argument (node).
        node: The root node of the pytree to map over
        with_path: If True, the function will be called with both the node and its path
        path_prefix: Optional prefix for the path. Used internally for recursion
        is_leaf: Optional function to determine if a node should be treated as a leaf,
            even if it's a pytree node. 
            If with_path is True, the function should take two arguments (node, path).
            Otherwise, it should take one argument (node).

            
    Returns:
        A new pytree with the function applied to each node. The structure of the tree
        is preserved, only the values are transformed.
    """
    if with_path:
        if not is_pytree_node(type(node)) or is_leaf(node, path_prefix):
            return fn(node, path_prefix)
    elif not is_pytree_node(type(node)) or is_leaf(node):
        return fn(node)

    children, aux, keys = flatten_once(node)
    if keys is None:
        keys = list(range(len(children)))
    key_tuples = [KeyTuple(type(node), key) for key in keys]
    mapped_children = [
        map(fn, child, with_path=with_path, path_prefix=path_prefix + [key_tuple], is_leaf=is_leaf)
        for child, key_tuple in zip(children, key_tuples)
    ]
    return unflatten_once(type(node), mapped_children, aux, keys)


def flatten_recursive(
    node: Any,
    with_path: bool = False,
    path_prefix: List[Any] = [],
    is_leaf: Callable[[Any], bool] = lambda *_: False,
) -> Tuple[List[Any], PyTreeDef, List[List[Any]]]:
    """Recursively flatten a pytree structure.
    
    This function converts a pytree into a flat list of leaves and a tree definition
    that can be used to reconstruct the original structure. It can optionally track
    the path to each leaf node.
    
    Args:
        node: The root node of the pytree to flatten
        with_path: If True, also return the paths to each leaf node
        path_prefix: Optional prefix for the path. Used internally for recursion
        is_leaf: Optional function to determine if a node should be treated as a leaf,
            even if it's a pytree node. 
            If with_path is True, the function should take two arguments (node, path).
            Otherwise, it should take one argument (node).
            
    Returns:
        A tuple containing:
        - List of leaf nodes
        - Tree definition that can be used to reconstruct the original structure
        - List of paths to each leaf node
    """
    if with_path:
        if not is_pytree_node(type(node)) or is_leaf(node, path_prefix):
            return [node], PyTreeDef(None, [], None, None), [path_prefix]
    elif not is_pytree_node(type(node)) or is_leaf(node):
        return [node], PyTreeDef(None, [], None, None), [path_prefix]

    children, aux, keys = flatten_once(node)
    if keys is None:
        keys = list(range(len(children)))
    key_tuples = [KeyTuple(type(node), key) for key in keys]
    leaves = []
    path_prefixes = []
    treedefs = []
    for child, key_tuple in zip(children, key_tuples):
        child_leaves, child_treedef, child_path_prefixes = flatten_recursive(
            child,
            with_path=with_path,
            path_prefix=path_prefix + [key_tuple],
            is_leaf=is_leaf,
        )
        leaves.extend(child_leaves)
        treedefs.append(child_treedef)
        path_prefixes.extend(child_path_prefixes)
    treedef = PyTreeDef(type(node), treedefs, aux, keys)
    return leaves, treedef, path_prefixes


def flatten(
    node: Any,
    with_path: bool = False,
    is_leaf: Callable[[Any], bool] = lambda *_: False,
) -> Tuple[List[Any], PyTreeDef, List[List[Any]]]:
    """Flatten a pytree structure into leaves and a tree definition.
    
    This is the main entry point for flattening pytrees. It converts a pytree into a flat
    list of leaves and a tree definition that can be used to reconstruct the original
    structure. It can optionally track the path to each leaf node.
    
    This function is a thin wrapper around flatten_recursive that handles the initial
    setup and ensures consistent return types regardless of whether with_path is True.
    
    Args:
        node: The root node of the pytree to flatten
        with_path: If True, also return the paths to each leaf node
        is_leaf: Optional function to determine if a node should be treated as a leaf,
            even if it's a pytree node. 
            If with_path is True, the function should take two arguments (node, path).
            Otherwise, it should take one argument (node).
            
    Returns:
        A tuple containing:
        - List of leaf nodes
        - Tree definition that can be used to reconstruct the original structure
        - List of paths to each leaf node
    """
    leaves, treedef, path_prefixes = flatten_recursive(
        node, with_path=with_path, is_leaf=is_leaf
    )

    if with_path:
        return leaves, treedef, path_prefixes
    else:
        return leaves, treedef


def unflatten_recursive(
    idx: int, leaves: List[Any], treedef: PyTreeDef, depth: int
) -> Any:
    """Recursively unflatten a pytree from leaves and a tree definition.
    
    Args:
        idx: Current index into the leaves list
        leaves: List of leaf nodes
        treedef: Tree definition
        depth: Current depth in the tree
        
    Returns:
        A tuple of (reconstructed_node, next_index)
    """
    if treedef.root is None:
        return leaves[idx], idx + 1

    children = []
    for child in treedef.children:
        child_node, idx = unflatten_recursive(idx, leaves, child, depth + 1)
        children.append(child_node)
    result = unflatten_once(treedef.root, children, treedef.aux, treedef.keys)
    if depth == 0:
        assert idx == len(leaves), "too many leaves in unflatten!"
    return result, idx


def unflatten(leaves: List[Any], treedef: PyTreeDef) -> Any:
    """Unflatten a pytree from leaves and a tree definition.
    
    Args:
        leaves: List of leaf nodes
        treedef: Tree definition
        
    Returns:
        The reconstructed pytree
    """
    return unflatten_recursive(0, leaves, treedef, 0)[0]


# register some default nodes


# list
def flatten_list(node: List[Any]) -> Tuple[List[Any], Any, Optional[List[Any]]]:
    """Flatten a list node.
    
    Args:
        node: List to flatten
        
    Returns:
        A tuple of (children, None, None)
    """
    return node, None, None


def unflatten_list(
    children: List[Any], aux: Any, keys: Optional[List[Any]]
) -> List[Any]:
    """Unflatten a list node.
    
    Args:
        children: List of child nodes
        aux: Unused
        keys: Unused
        
    Returns:
        The reconstructed list
    """
    return list(children)


register_pytree_node(list, flatten_list, unflatten_list)


# dict
def flatten_dict(node: Dict[Any, Any]) -> Tuple[List[Any], Any, Optional[List[Any]]]:
    """Flatten a dict node.
    
    Args:
        node: Dict to flatten
        
    Returns:
        A tuple of (values, None, keys)
    """
    return list(node.values()), None, list(node.keys())


def unflatten_dict(
    children: List[Any], aux: Any, keys: Optional[List[Any]]
) -> Dict[Any, Any]:
    """Unflatten a dict node.
    
    Args:
        children: List of values
        aux: Unused
        keys: List of keys
        
    Returns:
        The reconstructed dict
    """
    return dict(zip(keys, children))


register_pytree_node(dict, flatten_dict, unflatten_dict)


# tuple
def flatten_tuple(node: Tuple[Any, ...]) -> Tuple[List[Any], Any, Optional[List[Any]]]:
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
        keys = None
    return list(node), None, keys


def unflatten_tuple(
    children: List[Any], aux: Any, keys: Optional[List[Any]]
) -> Tuple[Any, ...]:
    """Unflatten a tuple node.
    
    Args:
        children: List of elements
        aux: Unused
        keys: Unused
        
    Returns:
        The reconstructed tuple
    """
    return tuple(children)


register_pytree_node(tuple, flatten_tuple, unflatten_tuple)

__all__ = [
    "map",
    "flatten",
    "unflatten",
    "KeyTuple",
    "PyTreeDef",
    "register_pytree_node",
]