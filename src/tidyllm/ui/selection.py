"""Interactive table selection UI using questionary."""

from collections.abc import Callable
from typing import Any, TypeVar

import questionary
from rich.console import Console
from rich.table import Table as RichTable

from tidyllm.context import get_tool_context
from tidyllm.types.linq import Table

T = TypeVar("T")


def select_ui(
    table: Table[T],
    title: str = "Select Items",
    display_columns: list[str] | None = None,
    multi_select: bool = True,
    formatter: dict[str, Callable] | None = None,
    show_preview: bool = True,
    compact: bool = False,
    pre_selected: list[int] | None = None,
) -> Table[T]:
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
        >>> data = Table.from_dict(
        ...     {"name": str, "age": int},
        ...     [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        ... )
        >>> selected = select_ui(data, title="Choose people")
    """
    console = get_tool_context().console

    # Check empty table
    if len(table) == 0:
        console.print("[yellow]No items to select[/yellow]")
        return Table.empty()

    # Determine columns
    if display_columns:
        columns_to_show = display_columns
    elif table.table_schema:
        columns_to_show = list(table.table_schema().model_fields.keys())
    else:
        # Fallback: try to infer from first row
        if table.rows and hasattr(table.rows[0], "__dict__"):
            columns_to_show = list(table.rows[0].__dict__.keys())
        else:
            columns_to_show = []

    # Show preview if requested
    if show_preview and not compact:
        _show_preview_table(table, title, columns_to_show, formatter, console)

    # Build choices
    choices = _build_choices(table, columns_to_show, formatter, compact, console)

    # Show interactive selector
    try:
        selected_indices = _run_selector(
            choices, 
            title if compact else "Make your selection:",
            multi_select,
            pre_selected
        )

        if selected_indices is None:
            return Table.empty()

        # Build result
        selected_rows = [table[i] for i in selected_indices]

        # Show summary
        if not compact:
            console.print(f"\n[green]✓ Selected {len(selected_rows)} item(s)[/green]")

        return Table(table_schema=table.table_schema, rows=selected_rows)

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Selection cancelled[/yellow]")
        return Table.empty()


def _show_preview_table(
    table: Table[T],
    title: str,
    columns: list[str],
    formatter: dict[str, Callable] | None,
    console: Console,
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
    formatter: dict[str, Callable] | None,
    compact: bool,
    console: Console,
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
    formatter: dict[str, Callable] | None
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
    pre_selected: list[int] | None
) -> list[int] | None:
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
