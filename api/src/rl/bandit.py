import random
from typing import List

# Bandit class with support for initial values
class Bandit:
    def __init__(self, num_arms: int, epsilon: float = 0.1, initial_values: List[float] = None):
        """Initialize the bandit with optional initial values for each arm."""
        self.num_arms = num_arms
        self.epsilon = epsilon
        if initial_values is None:
            self.values = [0.0] * num_arms
        else:
            if len(initial_values) != num_arms:
                raise ValueError("Length of initial_values must match num_arms")
            self.values = initial_values.copy()  # Avoid modifying the input list
        self.counts = [0] * num_arms

    def select_action(self) -> int:
        """Select an action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_arms - 1)
        return max(range(self.num_arms), key=lambda i: self.values[i])

    def update(self, action: int, reward: float):
        """Update the bandit's values based on the reward."""
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]