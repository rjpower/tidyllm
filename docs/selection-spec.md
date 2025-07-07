#Interactive Table Selection UI Specification

Overview
Build an interactive table selector using questionary that lets users navigate with arrow keys and select rows with spacebar. This replaces the clunky "type numbers" approach.
Dependencies
bashpip install questionary>=2.0.0 rich>=13.0.0
Core Function Specification
select_ui(table, **options) -> Table
Purpose: Show an interactive table selector and return selected rows.
Required Parameters:

table: Any object implementing the Table protocol

Optional Parameters:

title (str): Title shown above the selection. Default: "Select Items"
display_columns (list[str] | None): Which columns to show. Default: None (show all)
multi_select (bool): Allow multiple selections. Default: True
formatter (dict[str, Callable] | None): Format column values. Default: None
show_preview (bool): Show table preview before selection. Default: True
compact (bool): Use compact display format. Default: False
pre_selected (list[int] | None): Pre-select these indices. Default: None

Returns: New Table containing only selected rows
Keyboard Controls:

Multi-select mode:

↑/↓: Navigate between items
Space: Toggle selection
Enter: Confirm selections
Ctrl+A: Select all
Ctrl+R: Clear all selections
Ctrl+C or Esc: Cancel


Single-select mode:

↑/↓: Navigate between items
Enter: Select current item
Ctrl+C or Esc: Cancel



Implementation
pythonimport questionary
from typing import Optional, Callable
from rich.console import Console
from rich.table import Table as RichTable

console = Console()


def select_ui(
    table: Table,
    title: str = "Select Items",
    display_columns: Optional[list[str]] = None,
    multi_select: bool = True,
    formatter: Optional[dict[str, Callable]] = None,
    show_preview: bool = True,
    compact: bool = False,
    pre_selected: Optional[list[int]] = None,
) -> Table:
    """Interactive table selection using questionary.
    
    Shows an interactive selector where users can navigate with arrow keys
    and select items with spacebar (multi-select) or enter (single-select).
    
    Args:
        table: Input table to select from
        title: Title for the selection
        display_columns: Columns to display (None = all columns)
        multi_select: Enable multiple selection
        formatter: Dict of column_name -> format_function
        show_preview: Show full table before selection
        compact: Use compact format (no preview, minimal spacing)
        pre_selected: List of indices to pre-select
        
    Returns:
        New table containing only selected rows
        
    Example:
        >>> data = ConcreteTable.from_dict(
        ...     {"name": str, "age": int},
        ...     [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        ... )
        >>> selected = select_ui(data, title="Choose people")
    """
    # Check empty table
    if len(table) == 0:
        console.print("[yellow]No items to select[/yellow]")
        return ConcreteTable.empty()
    
    # Determine columns
    columns_to_show = display_columns or list(table.columns.keys())
    
    # Show preview if requested
    if show_preview and not compact:
        _show_preview_table(table, title, columns_to_show, formatter)
    
    # Build choices
    choices = _build_choices(table, columns_to_show, formatter, compact)
    
    # Show interactive selector
    try:
        selected_indices = _run_selector(
            choices, 
            title if compact else "Make your selection:",
            multi_select,
            pre_selected
        )
        
        if selected_indices is None:
            return ConcreteTable.empty()
        
        # Build result
        selected_rows = [table[i] for i in selected_indices]
        
        # Show summary
        if not compact:
            console.print(f"\n[green]✓ Selected {len(selected_rows)} item(s)[/green]")
        
        return ConcreteTable(columns=table.columns, rows=selected_rows)
        
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Selection cancelled[/yellow]")
        return ConcreteTable.empty()


def _show_preview_table(
    table: Table,
    title: str,
    columns: list[str],
    formatter: Optional[dict[str, Callable]]
) -> None:
    """Display a rich table preview."""
    rich_table = RichTable(title=title)
    
    # Add columns
    rich_table.add_column("#", style="dim", width=4)
    for col in columns:
        rich_table.add_column(col, style="cyan")
    
    # Add rows
    for i, row in enumerate(table):
        row_data = [str(i)]
        for col in columns:
            value = _get_formatted_value(row, col, formatter)
            row_data.append(value)
        rich_table.add_row(*row_data)
    
    console.print(rich_table)
    console.print()


def _build_choices(
    table: Table,
    columns: list[str],
    formatter: Optional[dict[str, Callable]],
    compact: bool
) -> list[questionary.Choice]:
    """Build questionary Choice objects for each row."""
    choices = []
    
    for i, row in enumerate(table):
        if compact:
            # Compact format: "0: value1 | value2"
            parts = []
            for col in columns:
                value = _get_formatted_value(row, col, formatter)
                parts.append(value)
            display = f"{i}: {' | '.join(parts)}"
        else:
            # Verbose format: "name: Alice | age: 30"
            parts = []
            for col in columns:
                value = _get_formatted_value(row, col, formatter)
                parts.append(f"{col}: {value}")
            display = " | ".join(parts)
        
        choices.append(questionary.Choice(title=display, value=i))
    
    return choices


def _get_formatted_value(
    row: Any,
    column: str,
    formatter: Optional[dict[str, Callable]]
) -> str:
    """Get formatted value from row for given column."""
    # Extract raw value
    if hasattr(row, column):
        value = getattr(row, column)
    elif hasattr(row, 'get'):
        value = row.get(column, "")
    else:
        value = ""
    
    # Apply formatter if exists
    if formatter and column in formatter:
        return formatter[column](value)
    
    return str(value)


def _run_selector(
    choices: list[questionary.Choice],
    message: str,
    multi_select: bool,
    pre_selected: Optional[list[int]]
) -> Optional[list[int]]:
    """Run the questionary selector and return selected indices."""
    if multi_select:
        # Multi-select with checkboxes
        default = None
        if pre_selected:
            default = [c for c in choices if c.value in pre_selected]
        
        result = questionary.checkbox(
            message,
            choices=choices,
            default=default,
            instruction="(Space: select, Enter: confirm, Ctrl+A: all)"
        ).ask()
        
        return result if result is not None else None
    else:
        # Single select
        default = None
        if pre_selected and len(pre_selected) > 0:
            default = next((c for c in choices if c.value == pre_selected[0]), None)
        
        result = questionary.select(
            message,
            choices=choices,
            default=default,
            instruction="(Enter: select, ↑↓: navigate)"
        ).ask()
        
        return [result] if result is not None else None
Helper Functions
pythondef select_ui_compact(table: Table, **kwargs) -> Table:
    """Compact version of select_ui (no preview, minimal output)."""
    return select_ui(table, compact=True, show_preview=False, **kwargs)


def select_one(table: Table, **kwargs) -> Optional[Any]:
    """Select a single row and return it (not a table)."""
    kwargs['multi_select'] = False
    result = select_ui(table, **kwargs)
    
    if len(result) > 0:
        return result[0]
    return None


def confirm_selection(
    table: Table,
    message: str = "Confirm these selections?"
) -> bool:
    """Show selected items and ask for confirmation."""
    if len(table) == 0:
        return False
    
    # Show what was selected
    console.print(f"\n[bold]Selected {len(table)} items:[/bold]")
    for i, row in enumerate(table):
        console.print(f"  {i+1}. {row}")
    
    # Ask for confirmation
    return questionary.confirm(message, default=True).ask()
Usage Examples
Basic Multi-Select
python# Sample data
vocab = ConcreteTable.from_dict(
    {"word": str, "translation": str},
    [
        {"word": "hello", "translation": "こんにちは"},
        {"word": "goodbye", "translation": "さようなら"},
        {"word": "thanks", "translation": "ありがとう"},
    ]
)

# Interactive selection
selected = select_ui(vocab, title="Choose vocabulary to study")
Single Select with Formatting
python# Custom formatter
def format_difficulty(value):
    colors = {"easy": "[green]", "medium": "[yellow]", "hard": "[red]"}
    return f"{colors.get(value, '')}{value}[/]"

# Select one item
selected_word = select_one(
    vocab_table,
    title="Choose a word to practice",
    formatter={"difficulty": format_difficulty}
)
Compact Mode for Scripts
python# No preview, minimal output
selected = select_ui_compact(
    table,
    title="Quick select",
    display_columns=["name", "priority"]
)
With Pre-selection
python# Pre-select first and third items
selected = select_ui(
    table,
    title="Review items (pre-selected based on difficulty)",
    pre_selected=[0, 2]
)
Error Handling

Empty table: Shows warning message, returns empty table
User cancels (Esc/Ctrl+C): Shows cancellation message, returns empty table
Invalid formatter: Catches exception, uses string conversion
Missing columns: Uses empty string for missing values

Testing Checklist

 Empty table shows warning
 Single item table works correctly
 Multi-select allows multiple selections
 Single-select allows only one selection
 Formatters apply correctly
 Pre-selection works
 Ctrl+C cancellation works
 Preview table displays correctly
 Compact mode skips preview
 Missing columns handled gracefully

Integration with Existing Code
Replace this pattern:
python# OLD: Manual index entry
console.print(table)
selection = console.input("Enter indices (comma-separated): ")
indices = [int(x) for x in selection.split(",")]
With:
python# NEW: Interactive selection
selected = select_ui(table)