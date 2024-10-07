import copy
import itertools
import random
import math

from matplotlib import pyplot as plt
import numpy as np
from carbs import CARBS, CARBSParams, LinearSpace, ObservationInParam, Param

class Space:
    def __init__(self, min_val, max_val, sampling='linear', base=None, rounding_factor=None, suggestion=None):
        self.min = min_val
        self.max = max_val
        self.sampling = sampling
        self.base = base
        self.rounding_factor = rounding_factor
        self.suggestion = suggestion

    def sample(self, underlying):
        if self.sampling == 'linear':
            value = self.min + (self.max - self.min) * underlying
        elif self.sampling == 'log_neg':
            range = self.max - self.min
            abs_range = abs(range)
            exp_factor = self.base ** (underlying * math.log(abs_range + 1, self.base))
            normalized_factor = (exp_factor - 1) / (self.base ** math.log(abs_range + 1, self.base) - 1)
            value = self.min + range * normalized_factor
        elif self.sampling == 'log':
            value = self.min + (self.max - self.min) * (math.exp(underlying * math.log(self.max / self.min)) - 1) / (math.exp(math.log(self.max / self.min)) - 1)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

        if self.rounding_factor:
            value = round(value / self.rounding_factor) * self.rounding_factor

        return value

    def reverse_sample(self, value):
        if self.rounding_factor:
            value = round(value / self.rounding_factor) * self.rounding_factor

        if self.sampling == 'linear':
            underlying = (value - self.min) / (self.max - self.min)
        elif self.sampling == 'log_neg':
            range = self.max - self.min
            abs_range = abs(range)
            normalized_factor = (value - self.min) / range
            underlying = math.log((normalized_factor * (self.base ** math.log(abs_range + 1, self.base) - 1) + 1), self.base) / math.log(abs_range + 1, self.base)
        elif self.sampling == 'log':
            underlying = math.log((value - self.min) / (self.max - self.min) * (self.base ** math.log(self.max / self.min, self.base) - 1) + 1, self.base) / math.log(self.max / self.min, self.base)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

        return underlying

class DiscreteSpace(Space):
    def __init__(self, values):
        super().__init__(0, len(values) - 1, 'linear', rounding_factor=1)
        self.values = values

    def sample(self, underlying):
        index = super().sample(underlying)
        return self.values[int(index)]

    def reverse_sample(self, value):
        try:
            index = self.values.index(value)
        except ValueError:
            raise ValueError(f"Value {value} not found in discrete space")
        return super().reverse_sample(index)

class Sweep:
    def __init__(self, config, better_direction_sign=1):
        self.config = config
        self.spaces = self._extract_spaces(config)
        if self.spaces:
            self.param_spaces = self._create_carbs_param_spaces()
            self.carbs_params = CARBSParams(better_direction_sign=better_direction_sign, is_wandb_logging_enabled=False, resample_frequency=0)
            self.carbs = CARBS(self.carbs_params, self.param_spaces)
        else:
            self.param_spaces = None
            self.carbs_params = None
            self.carbs = None

    def __bool__(self):
        return bool(self.spaces)

    def _extract_spaces(self, config):
        spaces = {}
        self._recursive_extract_spaces(config, spaces, [])
        return spaces

    def _recursive_extract_spaces(self, config, spaces, path):
        if isinstance(config, dict):
            if config.get('parameter', False):
                if 'values' in config:
                    spaces[tuple(path)] = DiscreteSpace(config['values'])
                else:
                    spaces[tuple(path)] = Space(
                        config['min'],
                        config['max'],
                        config.get('sampling', 'linear'),
                        config.get('base'),
                        config.get('rounding_factor'),
                        config.get('suggestion')
                    )
            else:
                for key, value in config.items():
                    self._recursive_extract_spaces(value, spaces, path + [key])
        elif isinstance(config, list):
            for i, item in enumerate(config):
                self._recursive_extract_spaces(item, spaces, path + [i])

    def _create_carbs_param_spaces(self):
        param_spaces = []
        for path, space in self.spaces.items():
            param_name = '_'.join(map(str, path))
            param_spaces.append(Param(name=param_name, space=LinearSpace(min=0, max=1), search_center=0.5))
        return param_spaces

    def suggest(self):
        if not self.spaces:
            return copy.deepcopy(self.config)
        
        carbs_suggestion = self.carbs.suggest().suggestion
        suggestion = copy.deepcopy(self.config)
        for (path, space), (param_name, param_value) in zip(self.spaces.items(), carbs_suggestion.items()):
            self._set_nested(suggestion, path, space.sample(param_value))
        return suggestion

    def _set_nested(self, config, path, value):
        for key in path[:-1]:
            config = config[key]
        config[path[-1]] = value

    def extract_varying_keys(self, config):
        return extract_varying_keys(self.config, config)

    def observe(self, config, value, cost):
        if not self.spaces:
            return
        
        varying_keys = self.extract_varying_keys(config)
        observation = {}
        for path, observed_value in varying_keys.items():
            space = self.spaces[path]
            normalized_value = space.reverse_sample(observed_value)
            param_name = '_'.join(map(str, path))
            observation[param_name] = normalized_value
        self.carbs.observe(ObservationInParam(input=observation, output=value, cost=cost))

def extract_varying_keys(sweep_config, suggested_config):
    def extract_recursive(sweep, suggested, path=[]):
        if isinstance(sweep, dict) and sweep.get('parameter', False):
            return {tuple(path): suggested}
        if isinstance(sweep, dict) and isinstance(suggested, dict):
            result = {}
            for key in sweep:
                if key in suggested:
                    sub_result = extract_recursive(sweep[key], suggested[key], path + [key])
                    if sub_result:
                        result.update(sub_result)
            return result
        elif isinstance(sweep, list) and isinstance(suggested, list):
            result = {}
            for i, (sweep_item, suggested_item) in enumerate(zip(sweep, suggested)):
                sub_result = extract_recursive(sweep_item, suggested_item, path + [i])
                if sub_result:
                    result.update(sub_result)
            return result
        return {}

    return extract_recursive(sweep_config, suggested_config)

def embellish_config(flat_config):
    nested_config = {}
    
    for keys, value in flat_config.items():
        current = nested_config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return nested_config





def old_main():
    import tkinter as tk
    from tkinter import ttk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import math

    def objective_function(config):
        # Larger values are better
        lr = config['learning_rate']
        batch_size = config['batch_size']
        optimizer = config['optimizer']
        layer1_units = config['model']['layers'][0]['units']
        layer2_units = config['model']['layers'][1]['units']
        
        # Optimal learning rate around 0.001
        lr_component = -100 * (math.log10(lr) + 3)**2
        
        # Larger batch sizes generally better
        batch_component = batch_size / 32
        
        # Optimizer preference: adam > rmsprop > sgd
        optimizer_score = {'adam': 1.0, 'rmsprop': 0.8, 'sgd': 0.6}
        optimizer_component = optimizer_score[optimizer]
        
        # Larger layers generally better
        layer_component = (layer1_units / 64 + layer2_units / 32) / 2
        
        return lr_component + batch_component + optimizer_component + layer_component

    def cost_function(config):
        # Higher cost for larger batch sizes and larger layers
        batch_size = config['batch_size']
        layer1_units = config['model']['layers'][0]['units']
        layer2_units = config['model']['layers'][1]['units']
        
        return (batch_size / 32) * ((layer1_units / 64 + layer2_units / 32) / 2)

    sweep_config = {
        "useless": True,
        'learning_rate': {
            'parameter': True,
            'sampling': 'log',
            'min': 1e-4,
            'max': 1e-1,
            'base': 10
        },
        'batch_size': {
            'parameter': True,
            'sampling': 'linear',
            'min': 32,
            'max': 256,
            'rounding_factor': 32
        },
        'optimizer': {
            'parameter': True,
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'model': {
            'layers': [
                {
                    'units': {
                        'parameter': True,
                        'sampling': 'log',
                        'min': 64,
                        'max': 128,
                        'base': 2,
                        'rounding_factor': 64
                    }
                },
                {
                    'units': {
                        'parameter': True,
                        'sampling': 'linear',
                        'min': 32,
                        'max': 256,
                        'rounding_factor': 32
                    }
                }
            ]
        }
    }

    sweep = Sweep(sweep_config)

    root = tk.Tk()
    root.title("Sweep Visualization")

    fig = plt.Figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Layer 1 Units')
    ax.set_title('Sweep Results')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    best_output = float('-inf')
    best_config = None
    results = []

    scatter = ax.scatter([], [], [], c=[], cmap='viridis')
    best_point = ax.scatter([], [], [], color='red', s=100, marker='*', label='Best')

    def update():
        nonlocal best_output, best_config

        suggestion = sweep.suggest()
        output = objective_function(suggestion)
        cost = cost_function(suggestion)
        sweep.observe(config=suggestion, value=output, cost=cost)
        
        learning_rate = suggestion['learning_rate']
        batch_size = suggestion['batch_size']
        layer1_units = suggestion['model']['layers'][0]['units']
        
        results.append((learning_rate, batch_size, layer1_units, output))
        
        if output > best_output:
            best_output = output
            best_config = suggestion
        
        print(f"Iteration {len(results)}:")
        print("Full Suggested Configuration:")
        print(suggestion)
        print(f"Objective Value: {output:.4f}")
        print(f"Cost: {cost:.4f}")
        print(f"Current Best Objective Value: {best_output:.4f}")

        varying_keys = extract_varying_keys(sweep_config, suggestion)
        print("\nVarying Keys:")
        for path, value in varying_keys.items():
            print(f"{'.'.join(map(str, path))}: {value}")
        
        x, y, z, c = zip(*results)
        scatter._offsets3d = (x, y, z)
        scatter.set_array(np.array(c))
        
        best_lr = best_config['learning_rate']
        best_bs = best_config['batch_size']
        best_l1 = best_config['model']['layers'][0]['units']
        best_point._offsets3d = ([best_lr], [best_bs], [best_l1])
        
        ax.set_title(f'Sweep Results (Iteration {len(results)})')
        
        # Update axis limits
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        ax.set_zlim(min(z), max(z))
        
        # Force redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        print("\n" + "-"*50 + "\n")
        
        if len(results) < 100:  # Limit to 100 iterations
            root.after(100, update)  # Schedule the next update

    update_button = ttk.Button(root, text="Start", command=update)
    update_button.pack()

    root.mainloop()
    
def main():
    import numpy as np
    import matplotlib.pyplot as plt

    # Create spaces
    linear_space = Space(0, 999, sampling='linear')
    # log_space = Space(-5, 69, sampling='log_neg', base=10, rounding_factor=2)
    log_space = Space(5, 69, sampling='log', base=2, rounding_factor=2)
    log_old_space = Space(5, 69, sampling='log', base=100, rounding_factor=2)
    discrete_space = DiscreteSpace(['a', 'b', 'c', 'd', 'e'])

    # Create plots
    fig, axs = plt.subplots(4, 2, figsize=(20, 20))
    fig.suptitle('Visualization of Different Spaces and Reverse Sampling', fontsize=16)

    spaces = [linear_space, log_space, log_old_space, discrete_space]
    space_names = ['Linear', 'Log', 'Log Old', 'Discrete']

    for i, (space, name) in enumerate(zip(spaces, space_names)):
        if name in ['Linear', 'Log', 'Log Old']:
            x_vals = np.linspace(0, 1, 1000)
            y_vals = np.array([space.sample(val) for val in x_vals])
            
            axs[i, 0].plot(x_vals, y_vals)
            axs[i, 0].set_title(f'{name} Space')
            axs[i, 0].set_xlabel('Underlying Value')
            axs[i, 0].set_ylabel('Sampled Value')
            
            # Reverse sampling plot
            reverse_x = y_vals
            reverse_y = np.array([space.reverse_sample(val) for val in reverse_x])
            
            axs[i, 1].plot(reverse_x, reverse_y)
            axs[i, 1].set_title(f'{name} Space - Reverse Sampling')
            axs[i, 1].set_xlabel('Sampled Value')
            axs[i, 1].set_ylabel('Recovered Underlying Value')
            axs[i, 1].plot([min(reverse_x), max(reverse_x)], [0, 1], 'r--', label='Expected')
            axs[i, 1].legend()
            
        else:  # Discrete
            x_vals = np.linspace(0, 1, len(space.values))
            y_vals = [space.sample(val) for val in x_vals]
            
            axs[i, 0].scatter(x_vals, y_vals)
            axs[i, 0].set_title('Discrete Space')
            axs[i, 0].set_xlabel('Underlying Value')
            axs[i, 0].set_ylabel('Sampled Value')
            axs[i, 0].set_yticks(space.values)
            
            # Reverse sampling plot
            reverse_x = y_vals
            reverse_y = [space.reverse_sample(val) for val in reverse_x]
            
            axs[i, 1].scatter(reverse_x, reverse_y)
            axs[i, 1].set_title('Discrete Space - Reverse Sampling')
            axs[i, 1].set_xlabel('Sampled Value')
            axs[i, 1].set_ylabel('Recovered Underlying Value')
            axs[i, 1].set_xticks(space.values)
            axs[i, 1].set_yticks(x_vals)

    plt.tight_layout()
    plt.show()

def test():
    config = {
        'a': {
            "parameter": True,
            "sampling": "linear",
            "min": 5,
            "max": 25,
            "rounding_factor": 5
        }
    }
    sweep = Sweep(config)
    for i in range(10):
        print(f"{embellish_config(sweep.extract_varying_keys(sweep.suggest()))=}")

if __name__ == "__main__":
    test()