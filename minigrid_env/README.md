# MiniGrid Environment

This is a Python package that provides a MiniGrid environment for reinforcement learning experiments.

## Requirements

- Python >= 3.10, < 3.13
- gymnasium >= 1.1.1
- minigrid >= 2.3.0
- numpy >= 1.26.0

## Installation

```bash
poetry install
```

## Usage

### Activating the Environment

After installation, you can activate the virtual environment in one of two ways:

1. Using Poetry's run command (recommended):
```bash
poetry run python your_script.py
```

2. Activating the virtual environment directly:
```bash
# Mac/Linux
source $(poetry env info --path)/bin/activate

# Windows (PowerShell)
. $(poetry env info --path)/Scripts/activate
```

To deactivate the environment when you're done:
```bash
deactivate
```

### Running Your Code

Once the environment is activated, you can run your Python scripts directly:
```bash
python your_script.py
```

Or use Poetry's run command without activating:
```bash
poetry run python your_script.py
```

## Development

TBD - Add development instructions as the project develops.