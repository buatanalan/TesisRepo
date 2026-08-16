import numpy as np

class HistoryBuffer:
    def __init__(self, spklu_ids: list, window_size_15m: int = 24 * 4 * 7):
        """
        window_size_15m: how many 15-min slots to remember (default 7 days = 672)
        """
        self.spklu_ids = spklu_ids
        self.id_to_idx = {sid: i for i, sid in enumerate(spklu_ids)}
        self.n = len(spklu_ids)
        
        self.window_size = window_size_15m
        
        # Ring buffer for utilization (shape: [window_size, n_spklu])
        self.util_history = np.zeros((self.window_size, self.n))
        self.current_step = 0
        self.is_full = False
        
    def record_utilization(self, utils: dict):
        """utils: dict of {spklu_id: util_value}"""
        idx = self.current_step % self.window_size
        for sid, util in utils.items():
            if sid in self.id_to_idx:
                self.util_history[idx, self.id_to_idx[sid]] = util
                
        self.current_step += 1
        if self.current_step >= self.window_size:
            self.is_full = True
            
    def get_recent_utilization(self, steps: int = 96):
        """Get average utilization over the last `steps` (96 = 24h)"""
        if self.current_step == 0:
            return np.zeros(self.n)
            
        max_available = self.window_size if self.is_full else self.current_step
        n_steps = min(steps, max_available)
        
        end_idx = self.current_step % self.window_size
        
        if end_idx >= n_steps:
            slice_hist = self.util_history[end_idx - n_steps:end_idx, :]
        else:
            # Wrap around
            part1 = self.util_history[self.window_size - (n_steps - end_idx):, :]
            part2 = self.util_history[:end_idx, :]
            if len(part2) > 0:
                slice_hist = np.vstack([part1, part2])
            else:
                slice_hist = part1
                
        return np.mean(slice_hist, axis=0)
        
    @staticmethod
    def calculate_gini(util_array: np.ndarray) -> float:
        """Calculate Gini coefficient for a 1D array of utilizations"""
        array = np.clip(util_array, 0, None)
        if np.sum(array) == 0:
            return 0.0
            
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

    def get_gini_trend(self, steps: int = 96):
        """Calculate slope of Gini over the last `steps`"""
        max_available = self.window_size if self.is_full else self.current_step
        n_steps = min(steps, max_available)
        if n_steps < 2:
            return 0.0
            
        end_idx = self.current_step % self.window_size
        
        if end_idx >= n_steps:
            slice_hist = self.util_history[end_idx - n_steps:end_idx, :]
        else:
            part1 = self.util_history[self.window_size - (n_steps - end_idx):, :]
            part2 = self.util_history[:end_idx, :]
            if len(part2) > 0:
                slice_hist = np.vstack([part1, part2])
            else:
                slice_hist = part1
                
        ginis = np.array([self.calculate_gini(row) for row in slice_hist])
        
        # Simple linear regression (slope)
        x = np.arange(n_steps)
        x_mean = np.mean(x)
        y_mean = np.mean(ginis)
        numerator = np.sum((x - x_mean) * (ginis - y_mean))
        denominator = np.sum((x - x_mean)**2)
        
        if denominator == 0: return 0.0
        return numerator / denominator
