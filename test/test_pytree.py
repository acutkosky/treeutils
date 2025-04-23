import pytest
from typing import NamedTuple
from treeutils import map, flatten, unflatten, KeyTuple

# Test data structures
class Point(NamedTuple):
    x: float
    y: float

class Person(NamedTuple):
    name: str
    age: int
    address: dict

def test_map_basic():
    # Test basic mapping over a list
    lst = [1, 2, 3]
    result = map(lambda x: x * 2, lst)
    assert result == [2, 4, 6]

    # Test mapping over a dict
    d = {'a': 1, 'b': 2}
    result = map(lambda x: x * 2, d)
    assert result == {'a': 2, 'b': 4}

    # Test mapping over a tuple
    t = (1, 2, 3)
    result = map(lambda x: x * 2, t)
    assert result == (2, 4, 6)

    # Test mapping over a NamedTuple
    p = Point(1.0, 2.0)
    result = map(lambda x: x * 2, p)
    assert result == Point(2.0, 4.0)

def test_map_nested():
    # Test nested structures
    data = {
        'points': [Point(1.0, 2.0), Point(3.0, 4.0)],
        'numbers': (1, 2, 3),
        'person': Person('Alice', 30, {'street': 'Main St'})
    }
    
    result = map(lambda x: x * 2 if isinstance(x, (int, float)) else x, data)
    assert result == {
        'points': [Point(2.0, 4.0), Point(6.0, 8.0)],
        'numbers': (2, 4, 6),
        'person': Person('Alice', 60, {'street': 'Main St'})
    }

def test_map_with_path():
    def path_aware_fn(node, path):
        return f"{node} at {path}"

    # Test with simple structure
    lst = [1, 2, 3]
    result = map(path_aware_fn, lst, with_path=True)
    assert result == ['1 at [0]', 
                     '2 at [1]', 
                     '3 at [2]']

    # Test with nested structure
    data = {'a': [1, 2], 'b': 3}
    result = map(path_aware_fn, data, with_path=True)
    assert result == {
        'a': ['1 at [a, 0]',
              '2 at [a, 1]'],
        'b': '3 at [b]'
    }

    # test with with NamedTuple
    p = Point(1.0, 2.0)
    result = map(path_aware_fn, p, with_path=True)
    assert result == Point('1.0 at [x]', '2.0 at [y]')

def test_map_with_is_leaf():
    def is_leaf(node):
        return isinstance(node, int)

    # Test with is_leaf that stops at integers
    data = {'a': [1, 2], 'b': 3}
    result = map(lambda x: x * 2, data, is_leaf=is_leaf)
    assert result == {'a': [2, 4], 'b': 6}

    # Test with is_leaf that stops at lists
    def is_list_leaf(node):
        return isinstance(node, list)
    
    data = {'a': [1, 2], 'b': 3}
    result = map(lambda x: str(x), data, is_leaf=is_list_leaf)
    assert result == {'a': '[1, 2]', 'b': '3'}  # List is treated as leaf, not transformed

def test_flatten_basic():
    # Test flattening a list
    lst = [1, 2, 3]
    leaves, treedef = flatten(lst)
    assert leaves == [1, 2, 3]
    assert unflatten(leaves, treedef) == lst

    # Test flattening a dict
    d = {'a': 1, 'b': 2}
    leaves, treedef = flatten(d)
    assert set(leaves) == {1, 2}  # Order not guaranteed
    assert unflatten(leaves, treedef) == d

    # Test flattening a tuple
    t = (1, 2, 3)
    leaves, treedef = flatten(t)
    assert leaves == [1, 2, 3]
    assert unflatten(leaves, treedef) == t

    # Test flattening a NamedTuple
    p = Point(1.0, 2.0)
    leaves, treedef = flatten(p)
    assert leaves == [1.0, 2.0]
    assert unflatten(leaves, treedef) == p

def test_flatten_nested():
    # Test nested structures
    data = {
        'points': [Point(1.0, 2.0), Point(3.0, 4.0)],
        'numbers': (1, 2, 3),
        'person': Person('Alice', 30, {'street': 'Main St'})
    }
    
    leaves, treedef = flatten(data)
    reconstructed = unflatten(leaves, treedef)
    assert reconstructed == data

def test_flatten_with_path():
    # Test flattening with paths
    data = {
        'points': [Point(1.0, 2.0), Point(3.0, 4.0)],
        'numbers': (1, 2, 3)
    }
    
    leaves, treedef, paths = flatten(data, with_path=True)

    assert paths == [
        [KeyTuple(dict,'points'), KeyTuple(list,0), KeyTuple(Point, 'x')],
        [KeyTuple(dict,'points'), KeyTuple(list,0), KeyTuple(Point, 'y')],
        [KeyTuple(dict,'points'), KeyTuple(list,1), KeyTuple(Point, 'x')],
        [KeyTuple(dict,'points'), KeyTuple(list,1), KeyTuple(Point, 'y')],
        [KeyTuple(dict,'numbers'), KeyTuple(tuple,0)],
        [KeyTuple(dict,'numbers'), KeyTuple(tuple,1)],
        [KeyTuple(dict,'numbers'), KeyTuple(tuple,2)],
    ]
    

def test_flatten_with_is_leaf():
    def is_leaf(node):
        return isinstance(node, int)

    # Test with is_leaf that stops at integers
    data = {'a': [1, 2], 'b': 3}
    leaves, treedef = flatten(data, is_leaf=is_leaf)
    assert leaves == [1, 2, 3]  # List is not flattened because it's not a leaf
    reconstructed = unflatten(leaves, treedef)
    assert reconstructed == data

def test_unflatten_edge_cases():
    # Test unflattening with empty structures
    leaves, treedef = flatten([])
    assert unflatten(leaves, treedef) == []

    leaves, treedef = flatten({})
    assert unflatten(leaves, treedef) == {}

    leaves, treedef = flatten(())
    assert unflatten(leaves, treedef) == ()

    # Test unflattening with single element
    leaves, treedef = flatten([1])
    assert unflatten(leaves, treedef) == [1]

def test_map_flatten_roundtrip():
    # Test that map and flatten/unflatten work together
    data = {
        'points': [Point(1.0, 2.0), Point(3.0, 4.0)],
        'numbers': (1, 2, 3)
    }
    
    # First map
    mapped = map(lambda x: x * 2 if isinstance(x, (int, float)) else x, data)
    
    # Then flatten and unflatten
    leaves, treedef = flatten(mapped)
    reconstructed = unflatten(leaves, treedef)
    
    assert reconstructed == mapped 

def test_empty_structures():
    # Test empty list
    lst = []
    result = map(lambda x: x * 2, lst)
    assert result == []
    leaves, treedef = flatten(lst)
    assert leaves == []
    assert unflatten(leaves, treedef) == []

    # Test empty dict
    d = {}
    result = map(lambda x: x * 2, d)
    assert result == {}
    leaves, treedef = flatten(d)
    assert leaves == []
    assert unflatten(leaves, treedef) == {}

    # Test empty tuple
    t = ()
    result = map(lambda x: x * 2, t)
    assert result == ()
    leaves, treedef = flatten(t)
    assert leaves == []
    assert unflatten(leaves, treedef) == ()

    # Test empty NamedTuple
    class Empty(NamedTuple):
        pass
    e = Empty()
    result = map(lambda x: x * 2, e)
    assert result == Empty()
    leaves, treedef = flatten(e)
    assert leaves == []
    assert unflatten(leaves, treedef) == Empty()

def test_nested_empty_structures():
    # Test list containing empty structures
    data = [[], {}, (), [{}], [[]]]
    result = map(lambda x: x, data)
    assert result == [[], {}, (), [{}], [[]]]
    leaves, treedef = flatten(data)
    assert leaves == []
    assert unflatten(leaves, treedef) == data

    # Test dict with empty values
    data = {'a': [], 'b': {}, 'c': ()}
    result = map(lambda x: x, data)
    assert result == {'a': [], 'b': {}, 'c': ()}
    leaves, treedef = flatten(data)
    assert leaves == []
    assert unflatten(leaves, treedef) == data

    # Test nested empty structures
    data = {'a': [{'b': []}], 'c': {'d': ()}}
    result = map(lambda x: x, data)
    assert result == {'a': [{'b': []}], 'c': {'d': ()}}
    leaves, treedef = flatten(data)
    assert leaves == []
    assert unflatten(leaves, treedef) == data

def test_special_cases():
    # Test None values
    data = {'a': None, 'b': [None, None]}
    result = map(lambda x: x, data)
    assert result == {'a': None, 'b': [None, None]}
    leaves, treedef = flatten(data)
    assert leaves == [None, None, None]
    assert unflatten(leaves, treedef) == data

    # Test mixed types with empty structures
    data = {
        'empty_list': [],
        'empty_dict': {},
        'empty_tuple': (),
        'non_empty': [1, 2, 3],
        'nested_empty': {'a': [], 'b': [{}]}
    }
    result = map(lambda x: x * 2 if isinstance(x, int) else x, data)
    assert result == {
        'empty_list': [],
        'empty_dict': {},
        'empty_tuple': (),
        'non_empty': [2, 4, 6],
        'nested_empty': {'a': [], 'b': [{}]}
    }
    leaves, treedef = flatten(data)
    assert leaves == [1, 2, 3]
    assert unflatten(leaves, treedef) == data

def test_map_with_path_empty():
    def path_aware_fn(node, path):
        return f"{node} at {path}"

    # Test empty list with path
    lst = []
    result = map(path_aware_fn, lst, with_path=True)
    assert result == []

    # Test empty dict with path
    d = {}
    result = map(path_aware_fn, d, with_path=True)
    assert result == {}

    # Test empty tuple with path
    t = ()
    result = map(path_aware_fn, t, with_path=True)
    assert result == ()

def test_flatten_with_path_empty():
    # Test empty structures with path tracking
    data = {'a': [], 'b': {}, 'c': ()}
    leaves, treedef, paths = flatten(data, with_path=True)
    assert leaves == []
    assert paths == []

    # Test nested empty structures with path tracking
    data = {'a': [{'b': []}], 'c': {'d': ()}}
    leaves, treedef, paths = flatten(data, with_path=True)
    assert leaves == []
    assert paths == []

def test_is_leaf_empty():
    def is_leaf(node):
        return isinstance(node, (list, dict, tuple)) and len(node) == 0

    # Test with is_leaf that treats empty structures as leaves
    data = {'a': [], 'b': [1, 2], 'c': {}}
    result = map(lambda x: x, data, is_leaf=is_leaf)
    assert result == {'a': [], 'b': [1, 2], 'c': {}}

    # Test flattening with the same is_leaf function
    leaves, treedef = flatten(data, is_leaf=is_leaf)
    assert leaves == [[], 1, 2, {}]
    assert unflatten(leaves, treedef) == data

