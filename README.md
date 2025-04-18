# RL Agents

## Installation

This project uses Poetry for dependency management. To get started:

1. Make sure you have Poetry installed:

   ```bash
   # Install pipx if you haven't already
   brew install pipx
   pipx ensurepath
   
   # Install Poetry using pipx
   pipx install poetry
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rl-agents.git
   cd rl-agents
   ```

3. Install dependencies:

   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry env activate
   ```

## Development

To add new dependencies:

```bash
poetry add <dependency>
```

## Install Baby AI

https://github.com/Farama-Foundation/Minigrid?tab=readme-ov-file

pip install minigrid




## Running LLMRL agent on Oscar

Connect to Oscar

```bash
ssh ssh.ccv.brown.edu
```

Request GPU Node and start LLama server
```bash
interact -n 4 -m 32g -q gpu -g 1 -t 1:00:00
module load ollama
ollama serve
```

Start a new ssh connection and connect to the same node

Display current jobs:
```bash
myq
```

connect to existing node:
```bash
ssh gpuxxxx
```

You can now run scripts that use the LLM agent:

```bash
python3 
```

