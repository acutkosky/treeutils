# TreeUtils

A Python library for working with tree-like data structures, providing utilities for flattening, unflattening, and mapping operations on nested Python objects. This library is modeled after JAX's pytree functionality but implemented as a standalone package without JAX dependencies.

## Installation

You can install TreeUtils directly from GitHub using pip:

```bash
pip install git+https://github.com/acutkosky/treeutils.git
```

## Features

- Flatten and unflatten nested Python objects (lists, dictionaries, tuples, and custom types)
- Map functions over tree structures while preserving the original structure
- Support for custom tree node types through registration
- Path tracking for tree traversal
- Built-in support for common Python types (list, dict, tuple)

## Usage Examples

### Basic Flattening and Unflattening

```python
from treeutils import flatten, unflatten

# Flatten a nested structure
nested = {
    'a': [1, 2, 3],
    'b': {'x': 4, 'y': 5}
}
leaves, treedef = flatten(nested)
print(leaves)  # [1, 2, 3, 4, 5]

# Reconstruct the original structure
reconstructed = unflatten(leaves, treedef)
print(reconstructed)  # {'a': [1, 2, 3], 'b': {'x': 4, 'y': 5}}
```

### Mapping Over Trees

```python
from treeutils import map

# Double all numbers in a nested structure
def double(x):
    return x * 2 if isinstance(x, (int, float)) else x

nested = {
    'a': [1, 2, 3],
    'b': {'x': 4, 'y': 5}
}
result = map(double, nested)
print(result)  # {'a': [2, 4, 6], 'b': {'x': 8, 'y': 10}}
```

### Custom Tree Node Types

```python
from treeutils import register_pytree_node
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# Register Point as a tree node
def flatten_point(point):
    return [point.x, point.y], None, ['x', 'y']

def unflatten_point(children, aux, keys):
    return Point(children[0], children[1])

register_pytree_node(Point, flatten_point, unflatten_point)

# Now Point objects can be used in tree operations
points = [Point(1, 2), Point(3, 4)]
leaves, treedef = flatten(points)
print(leaves)  # [1, 2, 3, 4]
```

### PyTorch Module Support

TreeUtils provides built-in support for PyTorch modules, allowing you to map functions over module parameters, buffers, and submodules. This is particularly useful for operations like parameter initialization, optimization, and model manipulation.

```python
import torch
import torch.nn as nn
from treeutils import map

# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
        self.register_buffer('buffer', torch.randn(10))

# Initialize a network
net = SimpleNet()

# Double all parameters and buffers
def double_tensor(x):
    if isinstance(x, torch.Tensor):
        return x * 2
    return x

mapped_net = map(double_tensor, net)

# Verify the mapping
assert torch.equal(mapped_net.linear1.weight, net.linear1.weight * 2)
assert torch.equal(mapped_net.linear2.weight, net.linear2.weight * 2)
assert torch.equal(mapped_net.buffer, net.buffer * 2)

# Map with path tracking
def path_aware_fn(x, path):
    if isinstance(x, torch.Tensor):
        return f"{x.shape} at {path}"
    return x

mapped_with_path = map(path_aware_fn, net, with_path=True)
print(mapped_with_path.linear1.weight)  # "torch.Size([20, 10]) at [linear1, weight]"
```

### Path Tracking

```python
from treeutils import flatten

# Get paths to each leaf node
nested = {
    'a': [1, 2, 3],
    'b': {'x': 4, 'y': 5}
}
leaves, treedef, paths = flatten(nested, with_path=True)
for leaf, path in zip(leaves, paths):
    print(f"Value: {leaf}, Path: {path}")
```

## API Reference

### Main Functions

- `flatten(node, with_path=False, is_leaf=None)`: Flatten a tree structure into leaves and a tree definition
- `unflatten(leaves, treedef)`: Reconstruct a tree from leaves and a tree definition
- `map(fn, node, with_path=False, is_leaf=None)`: Apply a function to each node in a tree
- `register_pytree_node(cls, flatten_fn, unflatten_fn)`: Register a new type as a tree node

### Classes

- `PyTreeDef`: Represents the structure of a tree
- `KeyTuple`: Represents a key in a tree path

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

