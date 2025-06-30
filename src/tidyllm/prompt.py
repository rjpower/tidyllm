"""Prompt loading with include directive support."""

import glob
import re
from pathlib import Path

import click


def module_dir(file_path: str) -> Path:
    """Get the directory containing a module file.

    Usage:
        @register(doc=read_prompt(module_dir(__file__) / "prompt.md"))
    """
    return Path(file_path).parent


def process_includes(text: str, current_path: Path, search_paths: list[Path]) -> str:
    """Process include directives in text."""
    pattern = r"\{\{include:\s*([^}]+)\}\}"

    def replace_include(match):
        include_path_str = match.group(1).strip()

        # Check if it's a glob pattern
        if '*' in include_path_str or '?' in include_path_str or '[' in include_path_str:
            # Handle glob patterns
            matched_files = []
            for search_path in search_paths:
                glob_pattern = str(search_path / include_path_str)
                matched_files.extend(glob.glob(glob_pattern))
            
            if not matched_files:
                # Fallback to current_path for backward compatibility
                glob_pattern = str(current_path / include_path_str)
                matched_files.extend(glob.glob(glob_pattern))
            
            if not matched_files:
                raise FileNotFoundError(
                    f"No files found matching glob pattern: {include_path_str} in any of the search paths"
                )
            
            # Sort files for consistent ordering
            matched_files.sort()
            
            # Read and concatenate all matched files
            combined_content = []
            for file_path in matched_files:
                file_path_obj = Path(file_path)
                file_content = file_path_obj.read_text()
                # Add filename guard
                combined_content.append(f"<file name=\"{file_path_obj.name}\">\n{file_content}\n</file>")
            
            # Join all contents and recursively process includes
            full_content = "\n\n".join(combined_content)
            return process_includes(full_content, current_path, search_paths)
        else:
            # Handle single file includes
            include_path = None
            for search_path in search_paths:
                potential_path = search_path / include_path_str
                if potential_path.exists():
                    include_path = potential_path
                    break

            if include_path is None:
                # Fallback to current_path for backward compatibility
                include_path = current_path / include_path_str
                if not include_path.exists():
                    raise FileNotFoundError(
                        f"Include file not found: {include_path_str} in any of the search paths"
                    )

            included_content = include_path.read_text()
            # Add filename guard for single files too
            guarded_content = f"<file name=\"{include_path.name}\">\n{included_content}\n</file>"
            # Recursively process includes in the included file
            return process_includes(guarded_content, include_path.parent, search_paths)

    return re.sub(pattern, replace_include, text)


def read_prompt(path: str | Path, source_paths: list[str | Path] | None = None, variables: dict[str, str] | None = None) -> str:
    """
    Read a PROMPT.md file and process {{include:}} directives and variable substitutions.

    Args:
        path: Path to the prompt file
        source_paths: Optional list of additional paths to search for include files
        variables: Optional dictionary of variables for substitution (e.g., {'name': 'value'})

    Example:
        # Main prompt with includes and variables
        {{include: ./sub_prompt.md}}
        Module: {{module_name}}
    """
    base_path = Path(path).parent
    content = Path(path).read_text()
    return expand_prompt(content, base_path, source_paths, variables)


def expand_prompt(
    prompt: str,
    base_path: Path,
    source_paths: list[Path] | None = None,
    variables: dict[str, str] | None = None,
) -> str:
    """
    Read a PROMPT.md file and process {{include:}} directives and variable substitutions.

    Args:
        prompt: Prompt content
        base_path: Base path for include file search
        source_paths: Optional list of additional paths to search for include files
        variables: Optional dictionary of variables for substitution (e.g., {'name': 'value'})

    Example:
        # Main prompt with includes and variables
        {{include: ./sub_prompt.md}}
        Module: {{module_name}}
    """
    # Add source paths to search directories
    search_paths = [base_path]
    if source_paths:
        search_paths.extend([Path(sp) for sp in source_paths])

    processed_content = process_includes(prompt, base_path, search_paths)

    # Handle escaped template variables first ({{{{var}}}} -> {{var}})
    processed_content = re.sub(r'\{\{\{\{([^}]+)\}\}\}\}', r'{{\1}}', processed_content)

    # Apply variable substitutions if provided
    if variables:
        for key, value in variables.items():
            processed_content = processed_content.replace(f"{{{{{key}}}}}", value)

    # Check for unspecified template variables
    unspecified_vars = re.findall(r'\{\{([^}]+)\}\}', processed_content)
    if unspecified_vars:
        unique_vars = set(unspecified_vars)
        raise ValueError(f"Unspecified template variables: {', '.join(sorted(unique_vars))}")

    return processed_content


@click.command()
@click.argument("prompt_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--source-paths",
    "-s",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Additional paths to search for include files",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (defaults to stdout)",
)
@click.option(
    "--variable",
    "-v",
    multiple=True,
    help="Variable substitution in format 'key=value'",
)
def resolve_prompt(
    prompt_file: Path, source_paths: tuple[Path, ...], output: Path | None, variable: tuple[str, ...]
):
    """
    Resolve a prompt.md file by processing all {{include:}} directives and variable substitutions.

    Example:
        python -m tidyllm.prompt prompt.md -s ./includes -o resolved_prompt.md -v module_name=chvalid -v version=1.0
    """
    try:
        # Parse variables from key=value format
        variables = {}
        for var in variable:
            if '=' not in var:
                raise ValueError(f"Invalid variable format: {var}. Expected 'key=value'")
            key, value = var.split('=', 1)
            variables[key] = value

        resolved_content = read_prompt(prompt_file, list(source_paths), variables)

        if output:
            output.write_text(resolved_content)
            click.echo(f"Resolved prompt written to: {output}")
        else:
            click.echo(resolved_content)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        raise click.Abort() from e


if __name__ == "__main__":
    resolve_prompt()
