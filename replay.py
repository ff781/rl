import torch
from torch.utils.data import Dataset, DataLoader
from tensordict import TensorDict
import random

class ReplayBuffer:
    def __init__(self, capacity=None):
        self.episodes = []
        self.capacity = capacity
        self.current_size = 0

    def add(self, data: TensorDict):
        # Ensure data is a TensorDict with shape (num_episodes, seq_len, ...)
        assert isinstance(data, TensorDict), "Data must be a TensorDict"
        assert len(data.batch_size) == 2, "Data must have shape (num_episodes, seq_len, ...)"
        
        new_episodes = data.unbind(0)
        new_size = sum(len(ep) for ep in new_episodes)

        if self.capacity is not None:
            # Calculate how much space we need to free up
            space_needed = self.current_size + new_size - self.capacity
            if space_needed > 0:
                # Remove oldest episodes until we have enough space
                cumsum = 0
                episodes_to_remove = 0
                for ep in self.episodes:
                    cumsum += len(ep)
                    episodes_to_remove += 1
                    if cumsum >= space_needed:
                        break
                
                self.episodes = self.episodes[episodes_to_remove:]
                self.current_size -= cumsum

        self.episodes.extend(new_episodes)
        self.current_size += new_size

    def get_dataloader(self, batches, bs, sl, mode=None):
        return DataLoader(
            ReplayDataset(self.episodes, batches, sl),
            batch_size=None,
            num_workers=0,
            collate_fn=collate_fn,
            sampler=BatchSampler(self.episodes, bs, batches, sl)
        )

class ReplayDataset(Dataset):
    def __init__(self, episodes, batches, seq_len):
        self.episodes = episodes
        self.batches = batches
        self.seq_len = seq_len

    def __len__(self):
        return self.batches

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self[i] for i in idx]
        episode = self.episodes[idx]
        start = random.randint(0, len(episode) - self.seq_len)
        r = episode[start:start+self.seq_len]
        return r

def collate_fn(batch):
    stacked = torch.stack(batch, dim=1)
    if torch.cuda.is_available():
        stacked = stacked.pin_memory()
        return stacked.to(torch.device('cuda'), non_blocking=True)
    return stacked

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, episodes, batch_size, num_batches, seq_len):
        self.episodes = episodes
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seq_len = seq_len

    def __iter__(self):
        for _ in range(self.num_batches):
            valid_episodes = [i for i, ep in enumerate(self.episodes) if len(ep) >= self.seq_len]
            yield random.choices(valid_episodes, k=self.batch_size)

    def __len__(self):
        return self.num_batches



def main():
    # Create a mock ReplayBuffer
    buffer = ReplayBuffer(capacity=1000)

    # Function to print current buffer size
    def print_buffer_size():
        real_size = sum(len(episode) for episode in buffer.episodes)
        print(f"Real buffer size: {real_size}")

    # Test adding data
    print("Adding data:")
    for i in range(30):
        mock_episode = TensorDict({
            'observation': torch.rand(2, 60, 10),
        }, batch_size=[2, 60])
        buffer.add(mock_episode)
        print_buffer_size()

    # Test adding more data to see if it respects capacity
    print("\nAdding more data:")
    for i in range(20):
        mock_episode = TensorDict({
            'observation': torch.rand(3, 70, 10),
        }, batch_size=[3, 70])
        buffer.add(mock_episode)
        print_buffer_size()

    # Test getting a dataloader
    print("\nTesting dataloader:")
    dataloader = buffer.get_dataloader(batches=5, bs=32, sl=50)
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        print(f"Batch keys: {batch.keys()}")
        break  # Just print the first batch

if __name__ == "__main__":
    main()
