# Offline Model-Based RL Library


## Introduction
Lightweight library that generalizes various offline model-based reinforcement learning (MBRL) algorithms into a single pipeline.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/offline-mbrl-library.git
   cd offline-mbrl-library
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the v-d4rl dataset:
   - Visit https://github.com/conglu1997/v-d4rl
   - Download the dataset
   - Create a `data` directory next to the project directory
   - Place the downloaded v-d4rl data in the `data` directory

## Usage

After installation, you can use the library by running `main.py`. Here are some usage examples:

1. Run a single training session OR create a wandb sweep from wandb sweep config format:
   ```
   python main.py --config config.json
   ```

2. Run a sweep of experiments:
   ```
   python main.py --sweep sweep_config.json
   ```

3. Run in test mode:
   ```
   python main.py --config config.json --is_test
   ```
