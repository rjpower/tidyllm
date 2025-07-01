#!/usr/bin/env python3
"""Package TidyLLM as a DXT (Desktop Extension) file using official DXT CLI."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_metadata() -> dict:
    """Extract metadata from pyproject.toml."""
    import tomllib
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    project = data.get("project", {})
    
    # Get version dynamically using git
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"], 
            capture_output=True, 
            text=True,
            cwd=pyproject_path.parent
        )
        version = result.stdout.strip() if result.returncode == 0 else "0.1.0"
    except (subprocess.SubprocessError, FileNotFoundError):
        version = "0.1.0"
    
    return {
        "name": project.get("name", "tidyllm"),
        "version": version,
        "description": project.get("description", ""),
        "author": project.get("authors", [{}])[0].get("name", ""),
        "email": project.get("authors", [{}])[0].get("email", ""),
        "repository": project.get("urls", {}).get("Repository", "")
    }


def update_manifest_version(manifest_path: Path, metadata: dict) -> None:
    """Update manifest.json with current version and metadata."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Update version and metadata
    manifest["version"] = metadata["version"]
    if metadata["description"]:
        manifest["description"] = metadata["description"]
    if metadata["author"]:
        manifest["author"]["name"] = metadata["author"]
    if metadata["email"]:
        manifest["author"]["email"] = metadata["email"]
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def create_temp_venv_package(repo_root: Path) -> Path:
    """Create a temporary directory with uv venv and project files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="tidyllm-dxt-"))
    print(f"📁 Created temporary directory: {temp_dir}")
    
    try:
        # Copy pyproject.toml to temp directory
        shutil.copy2(repo_root / "pyproject.toml", temp_dir / "pyproject.toml")
        print("📋 Copied pyproject.toml")
        
        # Copy source files
        if (repo_root / "src").exists():
            shutil.copytree(repo_root / "src", temp_dir / "src")
            print("📦 Copied src/ directory")
        
        # Copy other important files
        for file_name in ["manifest.json", "README.md", "LICENSE"]:
            src_file = repo_root / file_name
            if src_file.exists():
                shutil.copy2(src_file, temp_dir / file_name)
                print(f"📄 Copied {file_name}")
        
        # Install dependencies with uv
        print("⚙️  Installing dependencies with uv...")
        result = subprocess.run(
            ["uv", "sync"],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ uv sync failed: {result.stderr}")
            raise subprocess.SubprocessError(f"uv sync failed: {result.stderr}")
        
        # Verify python executable exists in venv
        venv_python_unix = temp_dir / ".venv" / "bin" / "python"
        venv_python_win = temp_dir / ".venv" / "Scripts" / "python.exe"
        
        if venv_python_unix.exists():
            # assert it's executable
            assert venv_python_unix.is_file() and venv_python_unix.stat().st_mode & 0o111
            print(f"✅ Found venv python at: {venv_python_unix}")
        elif venv_python_win.exists():
            print(f"✅ Found venv python at: {venv_python_win}")
        else:
            print("⚠️  Warning: Could not find python executable in venv")
        
        print("✅ Dependencies installed successfully")
        return temp_dir
        
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def package_with_dxt_cli(temp_dir: Path, output_name: str, final_output_dir: Path) -> bool:
    """Package using official DXT CLI from temporary directory."""
    try:
        # Validate manifest first
        print("🔍 Validating manifest...")
        result = subprocess.run(
            ["dxt", "validate", "manifest.json"],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ Manifest validation failed:")
            print(result.stderr)
            return False
        
        print("✅ Manifest is valid!")
        
        # Create temporary output path
        temp_output = temp_dir / output_name
        
        # Pack the DXT
        print("📦 Packaging with DXT CLI...")
        result = subprocess.run(
            ["dxt", "pack", ".", str(temp_output)],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ DXT packaging failed:")
            print(result.stderr)
            return False
        
        # Copy to final destination
        final_output_dir.mkdir(exist_ok=True)
        final_output_path = final_output_dir / output_name
        shutil.copy2(temp_output, final_output_path)
        
        print(result.stdout)
        print("\n✅ DXT package created successfully!")
        print("📦 Output: {final_output_path}")
        
        return True
        
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"❌ Error running DXT CLI: {e}")
        return False


def main():
    """Main packaging function."""
    repo_root = Path(__file__).parent.parent
    
    print("📦 TidyLLM DXT Packaging")
    print("=" * 40)
    
    metadata = extract_metadata()
    print(f"Package: {metadata['name']} v{metadata['version']}")
    
    # Create temporary venv package
    temp_dir = None
    try:
        temp_dir = create_temp_venv_package(repo_root)
        
        # Update manifest in temp directory
        temp_manifest_path = temp_dir / "manifest.json"
        update_manifest_version(temp_manifest_path, metadata)
        
        # Package with DXT CLI
        output_name = f"{metadata['name']}-{metadata['version']}.dxt"
        output_dir = repo_root / "dist"
        success = package_with_dxt_cli(temp_dir, output_name, output_dir)
        
    finally:
        # Clean up temporary directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("🧹 Cleaned up temporary directory")
    
    if success:
        print("\n🎉 Packaging completed successfully!")
        print("\n📋 Next steps:")
        print("1. Test the DXT locally")
        print("2. Install in Claude Desktop for testing")
        print(f"3. Share {output_name} with others")
    else:
        print("\n❌ Packaging failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()