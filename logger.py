import numpy as np
import wandb
import io
from PIL import Image

wandb.require("core")


class SimpleLogger:
    def __init__(self, project_name, config=None):
        self.run = None
        if project_name is not None:
            self.run = wandb.init(project=project_name, config=config)
        self.project_name = project_name
        self.config = config
        self.quantile_chart = QuantileChart()

    def log(self, data, step=None):
        import torch
        means_data = {k: v for k, v in data.items() if not isinstance(v, torch.Tensor)}
        quantiles_data = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}

        self.log_means(means_data, step)
        self.log_quantiles(quantiles_data, step)

    def log_means(self, data, step=None):
        import torch
        print(f"step {step:05d} (means):")
        for key, value in data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                1
        print()

        if self.run is not None:
            log_data = {}
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(img, Image.Image) for img in value):
                    # Convert list of PIL Images to numpy array
                    video_array = np.array([np.array(img) for img in value])
                    # Ensure the array has the correct shape (time, channel, height, width)
                    if video_array.ndim == 4:
                        video_array = np.transpose(video_array, (0, 3, 1, 2))
                    log_data[key] = wandb.Video(video_array, fps=5)
                elif torch.is_tensor(value) or isinstance(value, np.ndarray):
                    with torch.no_grad():
                        log_data[key] = value.mean().item()
                else:
                    log_data[key] = value
            
            wandb.log(log_data, step=step)

    def log_quantiles(self, data, step=None):
        import text
        print(f"step {step:05d} (quantiles):")
        print(text.std(data))

        if self.run is not None:
            log_data = {}
            self.quantile_chart.update(step, data)
            fig = self.quantile_chart.plot()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png')
            img_buf.seek(0)
            img = Image.open(img_buf)
            log_data["quantile_chart"] = wandb.Image(img)
            
            wandb.log(dict(quantile_chart=wandb.Image(img)), step=step)

    def finish(self):
        wandb.finish()

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class QuantileChart:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(list))
        self.steps = set()
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
            '#FFA500', '#FFC0CB', '#7FFFD4', '#8A2BE2', '#A52A2A', '#DEB887',
            '#5F9EA0', '#7FFF00'
        ]  # 20 visually distinct colors

    def update(self, step, data):
        self.steps.add(step)
        
        for key, value in data.items():
            if self._is_numeric(value):
                quantiles = self._calculate_quantiles(value)
                self.data[key][step] = quantiles

    def _is_numeric(self, value):
        import torch
        if np.isscalar(value):
            return True
        elif isinstance(value, (list, np.ndarray, torch.Tensor)):
            return True
        return False

    def _calculate_quantiles(self, value):
        if np.isscalar(value):
            return [value] * 5
        else:
            value = np.array(value).flatten()
            if len(value) == 0:
                return [np.nan] * 5  # Return NaN for empty lists
            else:
                return np.percentile(value, [0, 25, 50, 75, 100])

    def plot(self, partition_func=None):
        if partition_func is None:
            partition_func = lambda x: x

        partitions = defaultdict(list)
        for key in self.data.keys():
            partitions[partition_func(key)].append(key)

        n_plots = len(partitions)
        n_cols = min(3, n_plots)
        n_rows = (n_plots - 1) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
        fig.tight_layout(pad=4.0)

        for (partition, keys), ax in zip(partitions.items(), axes.flatten()):
            self._plot_partition(ax, partition, keys)

        for ax in axes.flatten()[len(partitions):]:
            ax.axis('off')

        return fig

    def _plot_partition(self, ax, partition, keys):
        steps = sorted(self.steps)
        for i, key in enumerate(keys):
            quantiles = []
            valid_steps = []
            for step in steps:
                if step in self.data[key]:
                    quantiles.append(self.data[key][step])
                    valid_steps.append(step)
            
            if not quantiles:
                continue

            quantiles = np.array(quantiles)
            color = self.colors[i % len(self.colors)]  # Cycle through colors
            
            ax.fill_between(valid_steps, quantiles[:, 0], quantiles[:, 1], alpha=0.1, color=color)
            ax.fill_between(valid_steps, quantiles[:, 1], quantiles[:, 3], alpha=0.2, color=color)
            ax.fill_between(valid_steps, quantiles[:, 3], quantiles[:, 4], alpha=0.1, color=color)
            ax.plot(valid_steps, quantiles[:, 2], label=key, color=color)

        ax.set_title(partition)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def main():
    # Create a QuantileChart instance
    chart = QuantileChart()

    # Generate some sample data
    steps = range(0, 1000, 100)
    for step in steps:
        data = {
            'loss': np.random.normal(10 - step/100, 2, 100),
            'accuracy': np.random.uniform(0.5 + step/2000, 1, 100),
            'learning_rate': 0.1 * (0.9 ** (step/100)),
            'metrics': dict(
                precision=np.random.beta(5 + step/200, 2, 100),
                recall=np.random.beta(3 + step/300, 2, 100),
            ),
            'gradients': dict(
                layer1=np.random.normal(0, 1 - step/1000, 100),
                layer2=np.random.normal(0, 0.8 - step/1250, 100),
            ),
        }
        chart.update(step, data)

    # Test basic plotting
    print("Plotting all data...")
    chart.plot()

    # Test plotting with custom partition function
    print("Plotting with custom partition function...")
    def custom_partition(key):
        if key.startswith('metrics'):
            return 'Metrics'
        elif key.startswith('gradients'):
            return 'Gradients'
        else:
            return 'Other'
    
    chart.plot(partition_func=custom_partition)

    # Test __call__ method
    print("Testing __call__ method...")
    chart.update(1000, {'new_metric': np.random.uniform(0, 1, 100)})

    # Verify that new data was added
    print("Plotting with new data...")
    chart.plot()

    # Test edge cases
    print("Testing edge cases...")
    chart.update(1100, {'single_value': 5})
    chart.update(1200, {'empty_list': []})
    chart.update(1300, {'single_item_list': [3]})
    chart.plot()

    print("QuantileChart testing complete.")

if __name__ == "__main__":
    main()

