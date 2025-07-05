"""Tests for object manipulation tools."""

import json
import pickle

import pytest

from tidyllm.tools.object import _parse_path, _traverse_path, get_attr, get_item


class TestPathParsing:
    """Test path parsing functionality."""
    
    def test_simple_field(self):
        """Test simple field access."""
        segments = _parse_path('.field')
        assert segments == ['field']
    
    def test_nested_field(self):
        """Test nested field access."""
        segments = _parse_path('.field.subfield')
        assert segments == ['field', 'subfield']
    
    def test_array_index(self):
        """Test array index access."""
        segments = _parse_path('.items[0]')
        assert segments == ['items', 0]
    
    def test_string_key(self):
        """Test string key access."""
        segments = _parse_path('.items["key"]')
        assert segments == ['items', 'key']
        
        segments = _parse_path(".items['key']")
        assert segments == ['items', 'key']
    
    def test_mixed_path(self):
        """Test mixed notation."""
        segments = _parse_path('.items[0].name')
        assert segments == ['items', 0, 'name']
        
        segments = _parse_path('.config["database"].host')
        assert segments == ['config', 'database', 'host']
    
    def test_root_array_access(self):
        """Test root array access."""
        segments = _parse_path('[0]')
        assert segments == [0]
    
    def test_invalid_path(self):
        """Test invalid path raises error."""
        with pytest.raises(ValueError):
            _parse_path('invalid')


class TestTraversePath:
    """Test path traversal functionality."""
    
    def test_dict_access(self):
        """Test dictionary access."""
        data = {'name': 'John', 'age': 30}
        result = _traverse_path(data, ['name'])
        assert result == 'John'
    
    def test_nested_dict_access(self):
        """Test nested dictionary access."""
        data = {'user': {'profile': {'name': 'John'}}}
        result = _traverse_path(data, ['user', 'profile', 'name'])
        assert result == 'John'
    
    def test_list_access(self):
        """Test list access."""
        data = [1, 2, 3]
        result = _traverse_path(data, [0])
        assert result == 1
    
    def test_mixed_access(self):
        """Test mixed dict/list access."""
        data = {'items': [{'name': 'test'}]}
        result = _traverse_path(data, ['items', 0, 'name'])
        assert result == 'test'
    
    def test_missing_key(self):
        """Test missing key raises error."""
        data = {'name': 'John'}
        with pytest.raises(KeyError):
            _traverse_path(data, ['missing'])
    
    def test_invalid_index(self):
        """Test invalid index raises error."""
        data = [1, 2, 3]
        with pytest.raises(IndexError):
            _traverse_path(data, [5])


class Testget_attr:
    """Test get_attr function."""
    
    def test_simple_access(self):
        """Test simple field access."""
        data = {'name': 'John', 'age': 30}
        result = get_attr('.name', input_data=data)
        assert result == 'John'
    
    def test_nested_access(self):
        """Test nested field access."""
        data = {'user': {'profile': {'name': 'John'}}}
        result = get_attr('.user.profile.name', input_data=data)
        assert result == 'John'
    
    def test_array_access(self):
        """Test array access."""
        data = {'items': [1, 2, 3]}
        result = get_attr('.items[0]', input_data=data)
        assert result == 1
    
    def test_mixed_access(self):
        """Test mixed notation."""
        data = {'items': [{'name': 'test'}]}
        result = get_attr('.items[0].name', input_data=data)
        assert result == 'test'
    
    def test_missing_field_strict(self):
        """Test missing field with strict mode."""
        data = {'name': 'John'}
        with pytest.raises(KeyError):
            get_attr('.missing', input_data=data)
    
    def test_json_input(self):
        """Test JSON string input."""
        json_data = '{"name": "John", "age": 30}'
        result = get_attr('.name', input_data=json.loads(json_data))
        assert result == 'John'
    
    def test_pickle_input(self):
        """Test pickle input."""
        data = {'name': 'John', 'age': 30}
        pickle.dumps(data)
        # Simulate what _read_stdin would do
        result = get_attr('.name', input_data=data)
        assert result == 'John'


class Testget_item:
    """Test get_item function."""
    
    def test_dict_access(self):
        """Test dictionary access."""
        data = {'name': 'John', 'age': 30}
        result = get_item('name', input_data=data)
        assert result == 'John'
    
    def test_list_access(self):
        """Test list access."""
        data = [1, 2, 3]
        result = get_item(0, input_data=data)
        assert result == 1
    
    
    def test_invalid_index(self):
        """Test invalid index."""
        data = [1, 2, 3]
        with pytest.raises(IndexError):
            get_item(5, input_data=data)