from typing import Callable, Type, Any, List, Tuple, NamedTuple

# A registry of pytree nodes
# pytree_node_registry[type] = (flatten_fn, unflatten_fn)
pytree_node_registry = {}


class RegistryEntry(NamedTuple):
    flatten_fn: Callable[[Any], Tuple[List[Any], Any, List[Any]]]
    unflatten_fn: Callable[[Any, List[Any], List[Any]], Any]


class KeyTuple(NamedTuple):
    type: Type
    key: Any

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
        keys: List[Any],
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
        children_string = [
            f"{key}: {child.as_string()}"
            for key, child in zip(self.keys, self.children)
        ]
        return f"{get_type_name(self.root)}[{', '.join(children_string)}]"

    def __repr__(self) -> str:
        """Get the string representation of the PyTreeDef.

        Returns:
            A string representation of the PyTreeDef
        """
        return f"PyTreeDef({self.as_string()})"


def register_pytree_node(
    cls: Type,
    flatten_fn: Callable[[Any], Tuple[List[Any], Any, List[Any]]],
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


def flatten_once(node: Any) -> Tuple[List[Any], Any, List[Any]]:
    """Flatten a single pytree node.

    Args:
        node: The node to flatten

    Returns:
        A tuple of (children, aux_data, keys)
    """
    return get_pytree_node_registry(type(node)).flatten_fn(node)


def unflatten_once(
    node_type: Any, children: List[Any], aux: Any, keys: List[Any]
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


def unzip(sequence):
    """Unzip a sequence of tuples into a tuple of sequences.

    Args:
        sequence: A sequence of tuples

    """
    return tuple(zip(*sequence))


def map(
    fn: Callable[[Any], Any],
    *tree: Any,
    with_path: bool = False,
    broadcast_prefix: bool = False,
    path_prefix: List[Any] = [],
    is_leaf: Callable[[Any], bool] = lambda *_: False,
) -> Any:
    """Map a function over one or more pytree structures.

    This function applies the given function to each node in the pytree(s), recursively traversing
    the tree structure. The function can optionally receive the path to each node.

    When mapping over multiple trees, all trees must have the same structure (same types and keys
    at each level), or the first tree's structure must be a prefix of the other trees' structures.

    If broadcast_prefix is True, then it is allowed for the later trees to have structure that is a
    prefix of the first tree's structure as well.

    The function will be called with corresponding leaf nodes from each tree.

    Args:
        fn: Function to apply to each node. If with_path is True, the function should accept
            N+1 arguments (node1, node2, ..., nodeN, path) where N is the number of trees.
            Otherwise, it should accept N arguments (node1, node2, ..., nodeN).
        tree: The root node(s) of the tree(s) to map over. The first tree's structure must be a prefix of
            the other trees' structures.
        with_path: If True, the function will be called with both the nodes and their path
        broadcast_prefix: If True, then it is allowed for the later trees to have structure that is a
            prefix of the first tree's structure. In this case, the leaf of tree[i] used as an argument to fn
        path_prefix: Optional prefix for the path. Used internally for recursion
        is_leaf: Optional function to determine if a node should be treated as a leaf,
            even if it's a pytree node.
            If with_path is True, the function should take two arguments (node, path).
            Otherwise, it should take one argument (node).


    Returns:
        A new pytree with the function applied to each node. The structure of the tree
        is preserved, only the values are transformed.

    Raises:
        ValueError: If the trees have different structures
    """
    if not tree:
        raise ValueError("At least one tree must be provided")

    # Check if we should treat this as a leaf
    def is_leaf_fn(node):
        is_leaf_args = (node,) if not with_path else (node, path_prefix)
        return not is_pytree_node(type(node)) or is_leaf(*is_leaf_args)

    if is_leaf_fn(tree[0]):
        if with_path:
            return fn(*tree, path_prefix)
        else:
            return fn(*tree)

    # Flatten the first tree
    children, aux, keys = flatten_once(tree[0])
    key_tuples = [KeyTuple(type(tree[0]), key) for key in keys]

    # Prepare children lists for all trees
    all_children = [[child] for child in children]

    # Flatten and verify structure of other trees
    if len(tree) > 1:
        for idx, other in enumerate(tree[1:], 1):
            if is_leaf_fn(other):
                if not broadcast_prefix:
                    raise ValueError(
                        f"Tree structures do not match at path {path_prefix}: encountered leaf node at input {idx}"
                    )
                for child_list in all_children:
                    child_list.append(other)
            else:
                other_children, other_aux, other_keys = flatten_once(other)

                if len(children) != len(other_children):
                    raise ValueError(
                        f"Tree structures do not match at path {path_prefix}: "
                        f"expected {len(children)} children, got {len(other_children)} at input {idx}"
                    )

                other_key_tuples = [KeyTuple(type(other), key) for key in other_keys]
                for key_tuple, other_key_tuple in zip(key_tuples, other_key_tuples):
                    if key_tuple.key != other_key_tuple.key:
                        raise ValueError(
                            f"Tree structures do not match at path {path_prefix}: "
                            f"expected key {key_tuple.key}, got {other_key_tuple.key} at input {idx}"
                        )

                # Add children from other tree
                for child_list, other_child in zip(all_children, other_children):
                    child_list.append(other_child)

    # Map over children
    mapped_children = [
        map(
            fn,
            *child_list,
            with_path=with_path,
            broadcast_prefix=broadcast_prefix,
            path_prefix=path_prefix + [key_tuple],
            is_leaf=is_leaf,
        )
        for child_list, key_tuple in zip(all_children, key_tuples)
    ]

    return unflatten_once(type(tree[0]), mapped_children, aux, keys)


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
        - List of paths to each leaf node (only if with_path is True)
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


__all__ = [
    "map",
    "flatten",
    "unflatten",
    "KeyTuple",
    "PyTreeDef",
    "register_pytree_node",
]
