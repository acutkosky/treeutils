import pytest
import torch
import torch.nn as nn
from treeutils import map, flatten, unflatten
import treeutils.pytorch_nodes


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer('buffer', torch.tensor([2.0]))
        self.submodule = nn.Linear(1, 1)
        self.submodule.weight.data.fill_(3.0)
        self.submodule.bias.data.fill_(4.0)

def test_flatten_module():
    # Create a simple module with parameters, buffers, and submodules
    module = SimpleModule()
    
    # Flatten the module
    leaves, treedef = flatten(module)
    
    # Check that we got all the leaves
    assert len(leaves) == 4  # param, buffer, submodule.weight, submodule.bias
    assert any(torch.equal(leaf, module.param) for leaf in leaves)
    assert any(torch.equal(leaf, module.buffer) for leaf in leaves)
    assert any(torch.equal(leaf, module.submodule.weight) for leaf in leaves)
    assert any(torch.equal(leaf, module.submodule.bias) for leaf in leaves)
    
    # Reconstruct and verify
    reconstructed = unflatten(leaves, treedef)
    assert isinstance(reconstructed, SimpleModule)
    assert torch.equal(reconstructed.param, module.param)
    assert torch.equal(reconstructed.buffer, module.buffer)
    assert torch.equal(reconstructed.submodule.weight, module.submodule.weight)
    assert torch.equal(reconstructed.submodule.bias, module.submodule.bias)

def test_map_module():
    # Create a simple module
    module = SimpleModule()
    
    # Map a function that doubles all tensors
    def double_tensor(x):
        if isinstance(x, torch.Tensor):
            return type(x)(x * 2)
        return x
    
    mapped = map(double_tensor, module)
    
    # Verify the mapping
    assert isinstance(mapped, SimpleModule)
    assert torch.equal(mapped.param, module.param * 2)
    assert torch.equal(mapped.buffer, module.buffer * 2)
    assert torch.equal(mapped.submodule.weight, module.submodule.weight * 2)
    assert torch.equal(mapped.submodule.bias, module.submodule.bias * 2)

def test_nested_modules():
    # Create a module with nested submodules
    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor([1.0]))
            self.submodule1 = SimpleModule()
            self.submodule2 = SimpleModule()
    
    module = NestedModule()
    
    # Flatten and verify
    leaves, treedef = flatten(module)
    assert len(leaves) == 9  # param + 4 leaves from each SimpleModule
    
    # Reconstruct and verify
    reconstructed = unflatten(leaves, treedef)
    assert isinstance(reconstructed, NestedModule)
    assert torch.equal(reconstructed.param, module.param)
    assert torch.equal(reconstructed.submodule1.param, module.submodule1.param)
    assert torch.equal(reconstructed.submodule2.param, module.submodule2.param)

def test_map_with_path():
    # Create a simple module
    module = SimpleModule()
    
    # Map with path tracking
    def path_aware_fn(x, path):
        if isinstance(x, torch.Tensor):
            return f"{x.item()} at {path}"
        return x
    
    mapped = map(path_aware_fn, module, with_path=True)
    
    # Verify the paths
    assert isinstance(mapped, SimpleModule)
    assert mapped.param == "1.0 at [param]"
    assert mapped.buffer == "2.0 at [buffer]"
    assert mapped.submodule.weight == "3.0 at [submodule, weight]"
    assert mapped.submodule.bias == "4.0 at [submodule, bias]"

def test_empty_module():
    # Test with a module that has no parameters, buffers, or submodules
    class EmptyModule(nn.Module):
        def __init__(self):
            super().__init__()
    
    module = EmptyModule()
    
    # Flatten and verify
    leaves, treedef = flatten(module)
    assert len(leaves) == 0
    
    # Reconstruct and verify
    reconstructed = unflatten(leaves, treedef)
    assert isinstance(reconstructed, EmptyModule)

def test_module_with_custom_attributes():
    # Test with a module that has custom attributes
    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor([1.0]))
            self.custom_attr = "not_a_tensor"
    
    module = CustomModule()
    
    # Flatten and verify
    leaves, treedef = flatten(module)
    assert len(leaves) == 1  # Only the parameter should be flattened
    assert torch.equal(leaves[0], module.param)
    
    # Reconstruct and verify
    reconstructed = unflatten(leaves, treedef)
    assert isinstance(reconstructed, CustomModule)
    assert torch.equal(reconstructed.param, module.param)
    assert reconstructed.custom_attr == module.custom_attr 

def test_map_two_modules():
    # Create two simple modules
    module1 = SimpleModule()
    module2 = SimpleModule()
    
    # Map a function that adds corresponding tensors
    def add_tensors(x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return type(x)(x + y)
        return x
    
    mapped = map(add_tensors, module1, module2)
    
    # Verify the mapping
    assert isinstance(mapped, SimpleModule)
    assert torch.equal(mapped.param, module1.param + module2.param)
    assert torch.equal(mapped.buffer, module1.buffer + module2.buffer)
    assert torch.equal(mapped.submodule.weight, module1.submodule.weight + module2.submodule.weight)
    assert torch.equal(mapped.submodule.bias, module1.submodule.bias + module2.submodule.bias)

def test_map_two_modules_with_path():
    # Create two simple modules
    module1 = SimpleModule()
    module2 = SimpleModule()
    
    # Map with path tracking
    def path_aware_add(x, y, path):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return f"({x.item()}+{y.item()}) at {path}"
        return x
    
    mapped = map(path_aware_add, module1, module2, with_path=True)
    
    # Verify the paths and values
    assert isinstance(mapped, SimpleModule)
    assert mapped.param == "(1.0+1.0) at [param]"
    assert mapped.buffer == "(2.0+2.0) at [buffer]"
    assert mapped.submodule.weight == "(3.0+3.0) at [submodule, weight]"
    assert mapped.submodule.bias == "(4.0+4.0) at [submodule, bias]"

def test_map_different_modules():
    # Create two different modules
    class DifferentModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor([5.0]))
            self.register_buffer('buffer', torch.tensor([6.0]))
            self.submodule = nn.Linear(1, 1)
            self.submodule.weight.data.fill_(7.0)
            self.submodule.bias.data.fill_(8.0)
    
    module1 = SimpleModule()
    module2 = DifferentModule()
    
    # Map a function that multiplies corresponding tensors
    def multiply_tensors(x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return type(x)(x * y)
        return x
    
    mapped = map(multiply_tensors, module1, module2)
    
    # Verify the mapping
    assert isinstance(mapped, SimpleModule)
    assert torch.equal(mapped.param, module1.param * module2.param)
    assert torch.equal(mapped.buffer, module1.buffer * module2.buffer)
    assert torch.equal(mapped.submodule.weight, module1.submodule.weight * module2.submodule.weight)
    assert torch.equal(mapped.submodule.bias, module1.submodule.bias * module2.submodule.bias)

def test_map_nested_modules():
    # Create two nested modules
    class NestedModule1(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor([1.0]))
            self.submodule = SimpleModule()
    
    class NestedModule2(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor([2.0]))
            self.submodule = SimpleModule()
    
    module1 = NestedModule1()
    module2 = NestedModule2()
    
    # Map a function that subtracts corresponding tensors
    def subtract_tensors(x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return type(x)(y - x)  # y - x to get positive values
        return x
    
    mapped = map(subtract_tensors, module1, module2)
    
    # Verify the mapping
    assert isinstance(mapped, NestedModule1)
    assert torch.equal(mapped.param, module2.param - module1.param)
    assert torch.equal(mapped.submodule.param, module2.submodule.param - module1.submodule.param)
    assert torch.equal(mapped.submodule.buffer, module2.submodule.buffer - module1.submodule.buffer)
    assert torch.equal(mapped.submodule.submodule.weight, module2.submodule.submodule.weight - module1.submodule.submodule.weight)
    assert torch.equal(mapped.submodule.submodule.bias, module2.submodule.submodule.bias - module1.submodule.submodule.bias)

def test_map_incompatible_modules():
    # Create modules with different structures
    class IncompatibleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor([1.0]))
            # Missing buffer and submodule
    
    module1 = SimpleModule()
    module2 = IncompatibleModule()
    
    # Attempt to map over incompatible modules
    with pytest.raises(ValueError):
        map(lambda x, y: x + y, module1, module2) 