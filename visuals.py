import os
import nets
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import math



def play_video(data, fps=60, grid_size=(3, 3)):

    # Coerce data to tensor if it's a numpy array
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif not isinstance(data, torch.Tensor):
        raise ValueError("Input must be either a numpy array or a PyTorch tensor")

    # Ensure data is in the correct shape (B, T, C, H, W)
    if data.dim() == 4:  # (T, C, H, W)
        data = data.unsqueeze(0)
    elif data.dim() != 5:
        raise ValueError("Input must have 4 or 5 dimensions")

    # Ensure data is in the correct shape (B, T, H, W, C) and permute to (B, T, C, H, W)
    if data.shape[2] == 3 or data.shape[2] == 1:  # If channels are in dimension 2
        data = data.permute(0, 1, 2, 3, 4)
    elif data.shape[4] == 3 or data.shape[4] == 1:  # If channels are in dimension 4
        data = data.permute(0, 1, 4, 2, 3)
    
    # Normalize data to [0, 255] range and convert to uint8
    if data.dtype != torch.uint8:
        data = (data - data.min()) / (data.max() - data.min()) * 255
        data = data.to(torch.uint8)

    # Calculate the number of videos per page
    videos_per_page = grid_size[0] * grid_size[1]

    def update_frame(frame_idx):
        if frame_idx < data.shape[1]:
            for i in range(videos_per_page):
                video_idx = selected_page.get() * videos_per_page + i
                if video_idx < data.shape[0]:
                    frame = data[video_idx, frame_idx].permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray(frame)
                    img = img.resize((video_width, video_height), Image.NEAREST)
                    imgtk = ImageTk.PhotoImage(image=img)
                    labels[i].imgtk = imgtk
                    labels[i].configure(image=imgtk)
            root.after(int(1000/fps), update_frame, frame_idx + 1)
        else:
            play_button.config(state=tk.NORMAL)

    def on_play():
        play_button.config(state=tk.DISABLED)
        update_frame(0)

    root = tk.Tk()
    root.title("Video Player")

    frame = ttk.Frame(root, padding="3")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Calculate window size based on video dimensions and grid size
    video_height, video_width = data.shape[3], data.shape[4]
    scale_factor = min(800 / (video_width * grid_size[1]), 600 / (video_height * grid_size[0]))
    video_width = int(video_width * scale_factor)
    video_height = int(video_height * scale_factor)
    window_width = video_width * grid_size[1] + 20  # Add some padding
    window_height = video_height * grid_size[0] + 100  # Add space for controls
    root.geometry(f"{window_width}x{window_height}")

    selected_page = tk.IntVar()
    page_selector = ttk.Combobox(frame, textvariable=selected_page, 
                                 values=list(range(math.ceil(data.shape[0] / videos_per_page))))
    page_selector.set(0)
    page_selector.grid(column=0, row=0, sticky=(tk.W, tk.E))

    play_button = ttk.Button(frame, text="Play", command=on_play)
    play_button.grid(column=1, row=0, sticky=(tk.W, tk.E))

    fps_var = tk.IntVar(value=fps)
    fps_entry = ttk.Entry(frame, textvariable=fps_var, width=5)
    fps_entry.grid(column=2, row=0, sticky=(tk.W, tk.E))
    ttk.Label(frame, text="FPS").grid(column=3, row=0, sticky=(tk.W, tk.E))

    labels = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            label = ttk.Label(frame)
            label.grid(column=j, row=i+1, padx=1, pady=1)
            labels.append(label)

    root.after(100, on_play)  # Start playback automatically after a short delay
    root.mainloop()

def plot_data_distributions(data, dirpath):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict
    
    # Prepare data for plotting
    plot_data = defaultdict(list)
    for key, value in data.items():
        plot_data[key].extend(value.flatten())
    
    # Create a new figure for each file
    plt.figure(figsize=(12, 8))
    
    # Plot distribution for each key
    for key, values in plot_data.items():
        if len(values) > 1:  # Only plot if we have more than one value
            sns.kdeplot(values, shade=True, label=key)
    
    plt.title(f"Value Distributions for {os.path.basename(dirpath)}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()



import torch
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from env import DMCEnv

def visualize_best_model(env=None, model=None, actor=None, critic=None, filepath=None):
    # Load the best model
    if filepath is not None:
        best_dict = torch.load(filepath, weights_only=False)
        env = DMCEnv(
            **best_dict['env']
        )
        model = best_dict['model']
        actor = best_dict['actor']
        critic = best_dict['critic']

    # Start a random episode
    done = False
    frames = []
    reconstructed_frames = []
    action_dists = []
    rewards = []
    predicted_rewards = []
    predicted_returns = []

    with torch.no_grad():

        obs, _ = model.reset()
        obs, _ = env.reset()

        while not done:
            # Encode observation and get action distribution
            markov_state = model.markov_state()
            action_dist = actor.get_dist(nets.cat(markov_state))
            
            # Sample action and step environment
            action = action_dist.sample().cpu().numpy()
            next_obs, reward, done, _, _ = env.step(action)
            _ = model.step(torch.as_tensor(action.copy(), device=next(model.parameters()).device))
            
            # Render and store frame
            frame = env.render()
            frames.append(frame)
            
            # Generate reconstructed observation from model
            reconstructed_obs = model.obs_decoder(markov_state)
            reconstructed_frame = model.postprocess_obs(reconstructed_obs).cpu().numpy().squeeze(0).transpose(1, 2, 0)
            reconstructed_frames.append(reconstructed_frame)

            # Store action distribution and reward
            action_dists.append(action_dist)
            rewards.append(reward)
            
            # Predict reward using world model
            predicted_reward,_ = model.reward_model(markov_state)
            predicted_reward = predicted_reward['reward'].item()
            predicted_rewards.append(predicted_reward)
            
            # Predict return using critic
            predicted_return = critic(nets.cat(markov_state)).item()
            predicted_returns.append(predicted_return)
            
            obs = next_obs
    
    # Create Tkinter window
    root = tk.Tk()
    root.title("Episode Visualization")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="3 3 12 12")
    main_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    
    # Create video frame
    video_frame = ttk.Frame(main_frame)
    video_frame.grid(column=0, row=0, columnspan=2)
    video_label = ttk.Label(video_frame)
    video_label.pack(side=tk.LEFT)
    reconstructed_video_label = ttk.Label(video_frame)
    reconstructed_video_label.pack(side=tk.RIGHT)
    
    # Create action distribution plots
    action_dim = env.action_space.shape[0]
    cols = min(3, action_dim)
    rows = (action_dim + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 2*rows), squeeze=False)
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=1, columnspan=2)
    
    # Create reward meter
    reward_var = tk.DoubleVar()
    reward_meter = ttk.Progressbar(main_frame, variable=reward_var, maximum=max(rewards))
    reward_meter.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E))
    ttk.Label(main_frame, text="Actual Reward").grid(column=2, row=2)
    
    # Create predicted reward meter
    predicted_reward_var = tk.DoubleVar()
    predicted_reward_meter = ttk.Progressbar(main_frame, variable=predicted_reward_var, maximum=max(predicted_rewards))
    predicted_reward_meter.grid(column=0, row=3, columnspan=2, sticky=(tk.W, tk.E))
    ttk.Label(main_frame, text="Predicted Reward (World Model)").grid(column=2, row=3)
    
    # Create predicted return meter
    predicted_return_var = tk.DoubleVar()
    predicted_return_meter = ttk.Progressbar(main_frame, variable=predicted_return_var, maximum=max(predicted_returns))
    predicted_return_meter.grid(column=0, row=4, columnspan=2, sticky=(tk.W, tk.E))
    ttk.Label(main_frame, text="Predicted Return (Critic)").grid(column=2, row=4)
    
    # Create FPS control
    fps_var = tk.IntVar(value=30)
    fps_label = ttk.Label(main_frame, text="FPS:")
    fps_label.grid(column=0, row=5)
    fps_entry = ttk.Entry(main_frame, textvariable=fps_var, width=5)
    fps_entry.grid(column=1, row=5)
    
    # Create step indicator
    step_var = tk.StringVar()
    step_label = ttk.Label(main_frame, textvariable=step_var)
    step_label.grid(column=0, row=6, columnspan=2)
    
    frame_index = 0
    current_fps = 30
    total_steps = len(frames)
    
    def update_display():
        nonlocal frame_index, current_fps
        
        # Update video frame
        img = Image.fromarray(frames[frame_index])
        img = img.resize((256, 256), Image.LANCZOS)  # Resize the image to be larger
        img = ImageTk.PhotoImage(img)
        video_label.config(image=img)
        video_label.image = img
        
        # Update reconstructed video frame
        reconstructed_img = Image.fromarray(reconstructed_frames[frame_index])
        reconstructed_img = reconstructed_img.resize((256, 256), Image.LANCZOS)
        reconstructed_img = ImageTk.PhotoImage(reconstructed_img)
        reconstructed_video_label.config(image=reconstructed_img)
        reconstructed_video_label.image = reconstructed_img
        
        # Update action distribution plots
        x = np.linspace(-1, 1, 100)
        x_expanded = np.tile(x[:, np.newaxis], (1, action_dim))
        y = action_dists[frame_index].log_prob(torch.tensor(x_expanded).to('cuda')).exp().detach().cpu().numpy()
        
        for i in range(action_dim):
            row = i // cols
            col = i % cols
            axs[row, col].clear()
            axs[row, col].plot(x, y[:, i])
            axs[row, col].set_title(f"Action Dim {i+1}")
            axs[row, col].set_ylabel("Probability Density")
        
        # Remove any unused subplots
        for i in range(action_dim, rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axs[row, col])
        
        fig.tight_layout()
        canvas.draw()
        
        # Update reward meters
        reward_var.set(rewards[frame_index])
        predicted_reward_var.set(predicted_rewards[frame_index])
        predicted_return_var.set(predicted_returns[frame_index])
        
        # Update step indicator
        step_var.set(f"Step: {frame_index + 1} / {total_steps}")
        
        frame_index = (frame_index + 1) % total_steps  # Loop around when we hit the end
        
        # Check if FPS has changed
        try:
            new_fps = fps_var.get()
            if new_fps != current_fps and new_fps > 0:
                current_fps = new_fps
        except tk.TclError:
            pass  # Invalid FPS value, keep using the current FPS
        
        root.after(int(1000/current_fps), update_display)
    
    update_display()
    root.mainloop()


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from tqdm import tqdm
import io
from PIL import Image
import torch

def visualize_best_model_static(env=None, model=None, actor=None, critic=None, filepath=None, device=None, max_len=100):
    # Determine the device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the best model
    if filepath is not None:
        best_dict = torch.load(filepath, map_location=device)
        env = DMCEnv(**best_dict['env'])
        model = best_dict['model']
        actor = best_dict['actor']
        critic = best_dict['critic']

    # Use the device context manager
    with torch.device(device):
        # Start a random episode
        done = False
        frames = []
        reconstructed_frames = []
        action_dists = []
        rewards = []
        predicted_rewards = []
        predicted_returns = []

        with torch.no_grad():
            obs, _ = model.reset()
            obs, _ = env.reset()

            for i in range(max_len):
                if done:
                    break
                markov_state = model.markov_state()
                action_dist = actor.get_dist(nets.cat(markov_state))
                
                action = action_dist.sample().cpu().numpy()
                next_obs, reward, done, _, _ = env.step(action)
                _ = model.step(torch.as_tensor(action.copy()))
                
                frame = env.render()
                frames.append(frame)
                
                reconstructed_obs = model.obs_decoder(markov_state)
                reconstructed_frame = model.postprocess_obs(reconstructed_obs).cpu().numpy().squeeze(0).transpose(1, 2, 0)
                reconstructed_frames.append(reconstructed_frame)

                action_dists.append(action_dist)
                rewards.append(reward)
                
                predicted_reward, _ = model.reward_model(markov_state)
                predicted_reward = predicted_reward['reward'].item()
                predicted_rewards.append(predicted_reward)
                
                predicted_return = critic(nets.cat(markov_state)).item()
                predicted_returns.append(predicted_return)
                
                obs = next_obs

    # Generate static images for each frame
    static_frames = []
    action_dim = env.action_space.shape[0]

    # Precompute action distribution data
    x = np.linspace(-1, 1, 100)
    x_expanded = np.tile(x[:, np.newaxis], (1, action_dim))
    
    with torch.device(device):
        y_data = [dist.log_prob(torch.tensor(x_expanded)).exp().cpu().numpy() for dist in action_dists]

    # Calculate global min and max for rewards
    all_values = rewards + predicted_rewards + predicted_returns
    global_min = min(all_values)
    global_max = max(all_values)
    # Iterate through each frame, using tqdm for a progress bar
    for frame_index in tqdm(range(min(max_len, len(frames))), desc="Generating static frames"):
        # Create a 2x3 grid of subplots with specific types and titles
        fig = make_subplots(rows=2, cols=3, 
                            specs=[[{'type': 'image'}, {'type': 'image'}, {'type': 'xy'}],
                                   [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]],
                            subplot_titles=("Original Frame", "Reconstructed Frame", "Action Distributions",
                                            "Rewards and Values"))

        # Add the original frame as an image to the first subplot
        fig.add_trace(go.Image(z=frames[frame_index]), row=1, col=1)
        
        # Add the reconstructed frame as an image to the second subplot
        fig.add_trace(go.Image(z=reconstructed_frames[frame_index]), row=1, col=2)
        
        # Create line plots for each action dimension in the third subplot
        for i in range(action_dim):
            fig.add_trace(go.Scatter(x=x, y=y_data[frame_index][:, i], name=f"{i+1}"), row=1, col=3)

        # Update layout for the action distribution plot
        # fig.update_xaxes(title_text="Action Value", row=1, col=3)
        # fig.update_yaxes(title_text="Probability Density", row=1, col=3)

        # Prepare data for the bar chart of rewards and predictions
        values = [rewards[frame_index], predicted_rewards[frame_index], predicted_returns[frame_index]]
        colors = ['blue', 'green', 'red']
        names = ['Env Reward', 'Model Reward', 'Critic']
        
        # Add a bar chart for rewards and predictions to the fourth subplot
        fig.add_trace(go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition='outside',
            textfont=dict(color='black'),
            width=0.5  # Reduce bar width
        ), row=2, col=1)

        # Set the y-axis range for the bar chart
        fig.update_yaxes(range=[global_min, global_max], title_text="Value", row=2, col=1)

        # Update the overall layout of the figure
        fig.update_layout(height=800, width=1200, title_text=f"Step: {frame_index + 1} / {len(frames)}")
        # Convert the figure to a PNG image
        img_bytes = fig.to_image(format="png")
        scale = 0.5
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size
        new_size = (int(width * scale), int(height * scale))
        img_resized = img.resize(new_size, Image.LANCZOS)
        # Add the downscaled PNG image to the list of static frames
        static_frames.append(img_resized)

    return static_frames

import imageio
from pathlib import Path
def save_frames_as_video(static_frames, output_path='video.mp4', fps=10):
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write frames to video file
    with imageio.get_writer(str(output_path), mode='I', fps=fps) as writer:
        for frame in static_frames:
            writer.append_data(np.array(frame))
    
    print(f"Looping video saved to {output_path}")



if __name__ == '__main__':

    # Example usage
    model_path = 'model.pt'
    static_frames = visualize_best_model_static(filepath=model_path)
    save_frames_as_video(static_frames, output_path='video.mp4')

    # visualize_best_model(filepath='model.pt')
