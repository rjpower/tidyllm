# CLI Parsing Bug Report

## Issue Summary

The CLI adapter in `src/tidyllm/adapters/cli.py` has a critical bug in argument parsing that prevents proper validation of function arguments when using named options with positional arguments.

## Bug Description

When using CLI commands with both positional arguments and named options, the parsing logic fails to correctly map the arguments to the function parameters, resulting in validation errors.

## Error Example

```bash
tidyllm get_attr --path='.test' --format=raw
```

Results in:
```
{"error": "Validation error: 1 validation error for Get_AttrArgs\npath\n  Field required [type=missing, input_value={'strict': False}, input_type=dict]"}
```

## Root Cause Analysis

The issue is in the `parse_cli_arguments()` function in `cli.py:124-147`. The function attempts to handle both positional arguments and named options, but the logic is flawed:

1. **Positional/Option Priority Logic**: The code tries to give positional arguments precedence over named options, but this creates confusion when only named options are provided.

2. **Missing Arguments**: When parsing `--path='.test'`, the function doesn't properly extract the value from the named option because it's looking for both a positional argument and a named option.

3. **Validation Input**: The validation receives `{'strict': False}` instead of the expected arguments with the `path` field.

## Current Faulty Logic

```python
# In parse_cli_arguments()
positional_value = kwargs.get(option.param_name)  # Gets None for --path
option_value = kwargs.get(click_param_name)       # Gets '.test' for --path
value = positional_value if positional_value is not None else option_value
```

The issue is that `kwargs.get(option.param_name)` for `path` returns `None` when using `--path='.test'`, and `kwargs.get(click_param_name)` should get the value but the mapping is incorrect.

## Specification for Correct CLI Parsing

### Requirements

1. **Dual Input Support**: Each function parameter should support both positional and named argument input
2. **Precedence Rules**: If both positional and named arguments are provided, positional takes precedence
3. **Proper Validation**: All arguments should be correctly mapped to function parameters for validation
4. **Error Handling**: Clear error messages when arguments are missing or invalid

### Expected Behavior

#### Single Argument Functions
```bash
# Both should work identically:
tidyllm get_attr '.test'
tidyllm get_attr --path='.test'
```

#### Multiple Argument Functions
```bash
# Mixed usage should work:
tidyllm some_function 'positional1' --arg2='named2' --arg3='named3'
```

#### JSON Input Override
```bash
# JSON input should bypass all other parsing:
tidyllm get_attr --json='{"path": ".test", "strict": false}'
```

### Implementation Requirements

1. **Click Parameter Mapping**: Ensure proper mapping between Click's parameter names (with underscores) and function parameter names
2. **Argument Collection**: Collect all provided arguments correctly, handling both positional and named forms
3. **Validation Integration**: Pass complete argument dictionary to Pydantic validation
4. **Default Handling**: Respect function parameter defaults and optional parameters

### Test Cases

The fixed implementation should pass these test cases:

```bash
# Basic positional argument
echo '{"test": "value"}' | tidyllm get_attr '.test'

# Basic named argument  
echo '{"test": "value"}' | tidyllm get_attr --path='.test'

# Mixed arguments
echo '{"test": "value"}' | tidyllm get_attr '.test' --strict=false

# JSON override
echo '{"test": "value"}' | tidyllm get_attr --json='{"path": ".test"}'

# All output formats
echo '{"test": "value"}' | tidyllm get_attr --path='.test' --format=json
echo '{"test": "value"}' | tidyllm get_attr --path='.test' --format=pickle
echo '{"test": "value"}' | tidyllm get_attr --path='.test' --format=raw
```

### Architecture Notes

- The CLI adapter should maintain backward compatibility with existing usage patterns
- Error messages should be clear and indicate which arguments are missing or invalid
- The implementation should handle Click's automatic hyphen-to-underscore conversion correctly
- All function parameters should be accessible via both positional and named arguments where appropriate

## Priority

**High** - This bug prevents basic CLI functionality and affects all tools that use the CLI adapter.

## Proposed Redesign

The current CLI generation approach has fundamental architectural flaws. Here's a clean redesign that addresses all the noted problems:

### Key Design Principles

1. **Separate Concerns**: Clearly separate Click command generation from argument parsing
2. **Simplified Architecture**: Remove the confusing dual positional/named argument approach
3. **Consistent Mapping**: Use a single, predictable mapping from CLI arguments to function parameters
4. **Clear Precedence**: Establish simple, understandable precedence rules

### Redesigned Architecture

#### 1. Command Generation Strategy

Instead of trying to create both positional and named arguments for every parameter, use a cleaner approach:

- **First Parameter Only**: Only the first parameter gets a positional argument
- **All Parameters**: All parameters get named options
- **Clear Precedence**: Positional argument (if provided) takes precedence over named option for first parameter

#### 2. New Implementation Structure

```python
def create_cli_command(func_desc: FunctionDescription) -> click.Command:
    """Generate CLI command with clean argument parsing."""
    
    @click.command(name=func_desc.name)
    @click.option("--json", "json_input", help="JSON input for all arguments")
    @click.option("--format", type=click.Choice(["json", "pickle", "raw"]), 
                  default="json", help="Output format")
    def cli(json_input: str | None, format: str, **kwargs):
        if json_input:
            args_dict = json.loads(json_input)
        else:
            args_dict = parse_cli_kwargs(kwargs, func_desc)
        
        # Validate and execute
        parsed_args = func_desc.validate_and_parse_args(args_dict)
        result = func_desc.call(**parsed_args)
        
        # Handle output format
        output_result(result, format)
    
    # Add CLI options dynamically
    cli = add_cli_options(cli, func_desc)
    return cli
```

#### 3. Simplified Option Generation

```python
def add_cli_options(cli_func: click.Command, func_desc: FunctionDescription) -> click.Command:
    """Add CLI options for function parameters."""
    fields = func_desc.args_model.model_fields
    
    # Add positional argument for first parameter only
    if fields:
        first_field_name = next(iter(fields))
        first_field = fields[first_field_name]
        
        cli_func = click.argument(
            first_field_name.upper(),  # Use uppercase for positional args
            required=False,
            type=get_click_type(first_field.annotation)
        )(cli_func)
    
    # Add named options for all parameters
    for field_name, field_info in fields.items():
        option_name = f"--{field_name.replace('_', '-')}"
        
        cli_func = click.option(
            option_name,
            type=get_click_type(field_info.annotation),
            help=field_info.description or f"Value for {field_name}",
            required=False  # Never required - we validate manually
        )(cli_func)
    
    return cli_func
```

#### 4. Clean Argument Parsing

```python
def parse_cli_kwargs(kwargs: dict[str, Any], func_desc: FunctionDescription) -> dict[str, Any]:
    """Parse CLI kwargs into function arguments with clear logic."""
    args_dict = {}
    fields = func_desc.args_model.model_fields
    
    # Handle first parameter specially (positional + named)
    if fields:
        first_field_name = next(iter(fields))
        
        # Check for positional argument (uppercase name)
        positional_value = kwargs.get(first_field_name.upper())
        
        # Check for named argument (with hyphen conversion)
        named_key = first_field_name.replace('_', '-')
        named_value = kwargs.get(named_key)
        
        # Positional takes precedence
        if positional_value is not None:
            args_dict[first_field_name] = positional_value
        elif named_value is not None:
            args_dict[first_field_name] = named_value
    
    # Handle remaining parameters (named only)
    for field_name in list(fields.keys())[1:]:  # Skip first parameter
        named_key = field_name.replace('_', '-')
        value = kwargs.get(named_key)
        
        if value is not None:
            args_dict[field_name] = value
    
    return args_dict
```

### Benefits of This Redesign

1. **Eliminates Confusion**: No more trying to handle both positional and named for every parameter
2. **Predictable Behavior**: Clear rules about what gets positional vs named arguments
3. **Simpler Logic**: Straightforward argument parsing without complex precedence rules
4. **Better UX**: More intuitive CLI interface that matches common CLI patterns
5. **Easier Testing**: Cleaner architecture makes testing much simpler

### Migration Strategy

1. **Backward Compatibility**: The new design maintains backward compatibility for existing usage
2. **Gradual Rollout**: Can be implemented incrementally with feature flags
3. **Clear Documentation**: Simple rules make it easy to document expected behavior

### Example Usage After Redesign

```bash
# Clean, intuitive usage patterns:
tidyllm get_attr '.test'                    # Positional for first param
tidyllm get_attr --path='.test'             # Named for first param  
tidyllm get_attr '.test' --strict=false     # Mixed: positional first, named others
tidyllm get_attr --path='.test' --strict=false --default='missing'  # All named
```

This redesign addresses all the root causes while providing a cleaner, more maintainable architecture.

## Files Affected

- `src/tidyllm/adapters/cli.py` - Primary fix location
- `tests/test_cli.py` - Test coverage needed
- All tools using CLI adapter - Indirect impact