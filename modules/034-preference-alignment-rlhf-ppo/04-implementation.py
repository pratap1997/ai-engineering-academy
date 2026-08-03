"""
Implementation of Preference Alignment (PPO vs DPO) algorithms from scratch.
"""

import math

def sigmoid(x):
    """Sigmoid function to map real values to probabilities."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

class PreferenceDataset:
    """A dataset of preference pairs: (prompt, win_response, lose_response)"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __iter__(self):
        return iter(self.data)

class BradleyTerryRewardModel:
    """
    A simple reward model that learns to score text responses.
    Uses Bradley-Terry model to predict preference probability.
    """
    def __init__(self):
        # A mock mapping from features/text to reward scores
        self.weights = {}
        self.learning_rate = 0.1
        
    def score(self, text):
        """Mock score based on length or specific words for illustration."""
        return self.weights.get(text, 0.0)
        
    def get_probability(self, win_text, lose_text):
        """Probability that win_text is preferred over lose_text."""
        r_win = self.score(win_text)
        r_lose = self.score(lose_text)
        return sigmoid(r_win - r_lose)
        
    def train_step(self, win_text, lose_text):
        """Update reward weights using gradient ascent on log-likelihood."""
        # Initialize if not present
        if win_text not in self.weights: self.weights[win_text] = 0.0
        if lose_text not in self.weights: self.weights[lose_text] = 0.0
            
        prob_win = self.get_probability(win_text, lose_text)
        
        # Gradient of log sigmoid(r_w - r_l): 
        # d/dr_w = 1 - P(win)
        # d/dr_l = P(win) - 1
        grad = 1.0 - prob_win
        
        self.weights[win_text] += self.learning_rate * grad
        self.weights[lose_text] -= self.learning_rate * grad

class KLController:
    """Manages the KL penalty to prevent policy from drifting too far."""
    def __init__(self, beta=0.1):
        self.beta = beta
        
    def compute_penalty(self, log_prob_policy, log_prob_ref):
        """KL divergence between policy and reference."""
        return self.beta * (log_prob_policy - log_prob_ref)

class PPOTrainer:
    """
    Simplified Proximal Policy Optimization for aligning a policy model.
    """
    def __init__(self, reward_model, kl_controller, clip_ratio=0.2):
        self.reward_model = reward_model
        self.kl_controller = kl_controller
        self.clip_ratio = clip_ratio
        
    def compute_loss(self, log_prob_old, log_prob_new, advantage):
        """Compute the clipped surrogate objective."""
        ratio = math.exp(log_prob_new - log_prob_old)
        
        obj1 = ratio * advantage
        obj2 = max(min(ratio, 1.0 + self.clip_ratio), 1.0 - self.clip_ratio) * advantage
        
        # We want to maximize this objective, so returning negative for a minimizer
        return -min(obj1, obj2)

class DPOLossCalculator:
    """
    Calculates the Direct Preference Optimization (DPO) loss.
    """
    def __init__(self, beta=0.1):
        self.beta = beta
        
    def compute_loss(self, log_prob_win_policy, log_prob_win_ref, 
                     log_prob_lose_policy, log_prob_lose_ref):
        """
        DPO Loss: -log sigmoid( beta * log(pi_theta(yw)/pi_ref(yw)) - beta * log(pi_theta(yl)/pi_ref(yl)) )
        """
        implicit_reward_win = self.beta * (log_prob_win_policy - log_prob_win_ref)
        implicit_reward_lose = self.beta * (log_prob_lose_policy - log_prob_lose_ref)
        
        diff = implicit_reward_win - implicit_reward_lose
        
        # We return the negative log likelihood
        # To avoid log(0), we bound the sigmoid output, or use log-sum-exp trick
        # log(sigmoid(x)) = x - log(1 + exp(x))
        if diff >= 0:
            loss = math.log(1.0 + math.exp(-diff))
        else:
            loss = -diff + math.log(1.0 + math.exp(diff))
            
        return loss
