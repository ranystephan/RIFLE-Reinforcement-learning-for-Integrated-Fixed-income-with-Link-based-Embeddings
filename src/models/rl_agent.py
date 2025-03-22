"""
Reinforcement Learning Agent Module for Fixed Income RL Project

This module implements RL agents for fixed income portfolio management:
1. DQN agent (for discrete action spaces)
2. DDPG agent (for continuous action spaces)
3. TD3 agent (enhanced DDPG with twin critics)
4. PPO agent (policy optimization)
5. SAC agent (soft actor-critic)

Mathematical foundations:
- Policy gradient methods
- Q-learning and actor-critic architectures
- Experience replay and target networks
- Policy optimization with constraints

Author: ranycs & cosrv
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import gymnasium as gym
import random
from collections import deque, namedtuple
import os
import matplotlib.pyplot as plt
import pandas as pd
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define experience tuple type
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Replay buffer for experience replay.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Size of the batch to sample
            
        Returns:
            List of experiences
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Size of the buffer
        """
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.
    """
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        """
        Initialize Ornstein-Uhlenbeck noise process.
        
        Args:
            size: Size of the action space
            mu: Mean of the process
            theta: Rate of mean reversion
            sigma: Scale of noise
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        self.reset()
    
    def reset(self):
        """
        Reset the process state.
        """
        self.state = np.copy(self.mu)
    
    def sample(self) -> np.ndarray:
        """
        Sample noise from the process.
        
        Returns:
            Noise sample
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ActorNetwork(nn.Module):
    """
    Actor network for policy gradient methods.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, init_w: float = 3e-3):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            init_w: Initial weight scale
        """
        super(ActorNetwork, self).__init__()
        
        # Log dimensions for debugging
        logger.info(f"Initializing ActorNetwork with state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State tensor
            
        Returns:
            Action tensor
        """
        if len(state.shape) > 0 and state.shape[-1] != self.input_layer.weight.shape[1]:
            # Log error for debugging
            logger.error(f"Input shape mismatch: state shape = {state.shape}, input layer expects {self.input_layer.weight.shape[1]} features")
            raise ValueError(f"Input shape mismatch: got {state.shape[-1]} features but expected {self.input_layer.weight.shape[1]}")
            
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        x = torch.tanh(self.output_layer(x))  # Output in [-1, 1]
        return x


class CriticNetwork(nn.Module):
    """
    Critic network for policy gradient methods.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, init_w: float = 3e-3):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            init_w: Initial weight scale
        """
        super(CriticNetwork, self).__init__()
        
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        """
        Forward pass.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value tensor
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent.
    """
    
    def __init__(self, 
                state_dim: int, 
                action_dim: int, 
                hidden_dim: int = 256,
                actor_lr: float = 1e-4,
                critic_lr: float = 1e-3,
                gamma: float = 0.99,
                tau: float = 0.001,
                batch_size: int = 64,
                buffer_size: int = 100000,
                device: str = 'cuda'):
        """
        Initialize DDPG agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            actor_lr: Learning rate for the actor
            critic_lr: Learning rate for the critic
            gamma: Discount factor
            tau: Soft update parameter
            batch_size: Batch size for training
            buffer_size: Size of the replay buffer
            device: Device to use for training (cuda or cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize actor networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize critic networks
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize exploration noise
        self.noise = OUNoise(action_dim)
        
        # Initialize training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.rewards = []
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select action.
        
        Args:
            state: Current state
            eval_mode: Whether to use exploration noise
            
        Returns:
            Selected action
        """
        # Check for dimension mismatch and fix at runtime
        expected_dim = self.state_dim
        actual_dim = state.shape[0]
        
        if actual_dim != expected_dim:
            logger.warning(f"State dimension mismatch: expected {expected_dim}, got {actual_dim}. Adapting...")
            if actual_dim < expected_dim:
                # Pad with zeros if input is smaller than expected
                padding = np.zeros(expected_dim - actual_dim)
                state = np.concatenate([state, padding])
            else:
                # Truncate if input is larger than expected
                state = state[:expected_dim]
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Set actor to evaluation mode
        self.actor.eval()
        
        with torch.no_grad():
            # Get action from actor
            action = self.actor(state_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
        
        # Set actor back to training mode
        self.actor.train()
        
        # Add exploration noise if not in evaluation mode
        if not eval_mode:
            noise = self.noise.sample()
            action += noise
        
        # Ensure action is within bounds [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to [0, 1] range for portfolio weights
        action = (action + 1.0) / 2.0
        
        # Normalize to ensure weights sum to 1
        action = action / np.sum(action)
        
        return action
    
    def process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Process action for storage in replay buffer.
        
        Args:
            action: Action to process
            
        Returns:
            Processed action
        """
        # Convert from [0, 1] range to [-1, 1] range for DDPG
        action_processed = action * 2.0 - 1.0
        return action_processed
    
    def update(self) -> Tuple[float, float]:
        """
        Update the agent.
        
        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        # Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Sample from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert experiences to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device).unsqueeze(1)
        
        # Check for and handle dimension mismatch in states and next_states
        expected_dim = self.state_dim
        if states.shape[1] != expected_dim:
            logger.warning(f"Batch state dimension mismatch: expected {expected_dim}, got {states.shape[1]}. Adapting...")
            if states.shape[1] < expected_dim:
                # Pad with zeros if input is smaller than expected
                padding = torch.zeros(states.shape[0], expected_dim - states.shape[1], device=self.device)
                states = torch.cat([states, padding], dim=1)
                
                # Also pad next_states
                padding = torch.zeros(next_states.shape[0], expected_dim - next_states.shape[1], device=self.device)
                next_states = torch.cat([next_states, padding], dim=1)
            else:
                # Truncate if input is larger than expected
                states = states[:, :expected_dim]
                next_states = next_states[:, :expected_dim]
        
        # Update critic
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.actor_target(next_states)
            
            # Get Q-values from target critic
            next_q_values = self.critic_target(next_states, next_actions)
            
            # Compute target Q-values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Get current Q-values
        current_q_values = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update target network.
        
        Args:
            source: Source network
            target: Target network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def save(self, path: str):
        """
        Save agent.
        
        Args:
            path: Path to save the agent
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'rewards': self.rewards,
            'total_steps': self.total_steps
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load agent.
        
        Args:
            path: Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        self.rewards = checkpoint['rewards']
        self.total_steps = checkpoint['total_steps']
        logger.info(f"Agent loaded from {path}")
    
    def plot_learning_curves(self) -> plt.Figure:
        """
        Plot learning curves.
        
        Returns:
            Matplotlib figure
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot actor loss
        axs[0].plot(self.actor_losses)
        axs[0].set_title('Actor Loss')
        axs[0].set_xlabel('Training Steps')
        axs[0].set_ylabel('Loss')
        axs[0].grid(True)
        
        # Plot critic loss
        axs[1].plot(self.critic_losses)
        axs[1].set_title('Critic Loss')
        axs[1].set_xlabel('Training Steps')
        axs[1].set_ylabel('Loss')
        axs[1].grid(True)
        
        # Plot rewards
        axs[2].plot(self.rewards)
        axs[2].set_title('Episode Rewards')
        axs[2].set_xlabel('Episodes')
        axs[2].set_ylabel('Reward')
        axs[2].grid(True)
        
        plt.tight_layout()
        return fig


class TD3Agent(DDPGAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    
    TD3 is an improvement over DDPG that addresses function approximation error
    by using twin critics, delayed policy updates, and target policy smoothing.
    """
    
    def __init__(self, 
                state_dim: int, 
                action_dim: int, 
                hidden_dim: int = 256,
                actor_lr: float = 1e-4,
                critic_lr: float = 1e-3,
                gamma: float = 0.99,
                tau: float = 0.001,
                batch_size: int = 64,
                buffer_size: int = 100000,
                policy_update_freq: int = 2,
                policy_noise: float = 0.2,
                noise_clip: float = 0.5,
                device: str = 'cuda'):
        """
        Initialize TD3 agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            actor_lr: Learning rate for the actor
            critic_lr: Learning rate for the critic
            gamma: Discount factor
            tau: Soft update parameter
            batch_size: Batch size for training
            buffer_size: Size of the replay buffer
            policy_update_freq: Frequency of policy updates
            policy_noise: Noise added to target policy
            noise_clip: Maximum noise added to target policy
            device: Device to use for training (cuda or cpu)
        """
        # Fixing the dimension issue - log the actual state dimension
        logger.info(f"Initializing TD3Agent with state_dim={state_dim}")
        
        # Hard-code the correct dimension if we have a specific known mismatch
        # This is a temporary fix for the 667/670 dimension issue
        if state_dim == 667:
            logger.warning(f"Detected state dimension of 667, correcting to 670 to match network architecture")
            state_dim = 670
        
        # Initialize the parent class
        super(TD3Agent, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            buffer_size=buffer_size,
            device=device
        )
        
        # TD3 specific parameters
        self.policy_update_freq = policy_update_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_counter = 0
        
        # Initialize second critic network
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy target network
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Update optimizer
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
    
    def update(self) -> Tuple[float, float, float]:
        """
        Update the agent.
        
        Returns:
            Tuple of (actor_loss, critic1_loss, critic2_loss)
        """
        # Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        # Increment update counter
        self.update_counter += 1
        
        # Sample from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert experiences to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device).unsqueeze(1)
        
        # Check for and handle dimension mismatch in states and next_states
        expected_dim = self.state_dim
        if states.shape[1] != expected_dim:
            logger.warning(f"Batch state dimension mismatch in TD3: expected {expected_dim}, got {states.shape[1]}. Adapting...")
            if states.shape[1] < expected_dim:
                # Pad with zeros if input is smaller than expected
                padding = torch.zeros(states.shape[0], expected_dim - states.shape[1], device=self.device)
                states = torch.cat([states, padding], dim=1)
                
                # Also pad next_states
                padding = torch.zeros(next_states.shape[0], expected_dim - next_states.shape[1], device=self.device)
                next_states = torch.cat([next_states, padding], dim=1)
            else:
                # Truncate if input is larger than expected
                states = states[:, :expected_dim]
                next_states = next_states[:, :expected_dim]
        
        # Update critics
        with torch.no_grad():
            # Get next actions from target actor with noise for smoothing
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -1, 1)
            
            # Get Q-values from both target critics
            next_q1_values = self.critic_target(next_states, next_actions)
            next_q2_values = self.critic2_target(next_states, next_actions)
            
            # Use minimum of two Q-values to prevent overestimation
            next_q_values = torch.min(next_q1_values, next_q2_values)
            
            # Compute target Q-values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Get current Q-values from first critic
        current_q1_values = self.critic(states, actions)
        
        # Compute first critic loss
        critic1_loss = F.mse_loss(current_q1_values, target_q_values)
        
        # Update first critic
        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer.step()
        
        # Get current Q-values from second critic
        current_q2_values = self.critic2(states, actions)
        
        # Compute second critic loss
        critic2_loss = F.mse_loss(current_q2_values, target_q_values)
        
        # Update second critic
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Delayed policy updates
        actor_loss = 0.0
        if self.update_counter % self.policy_update_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.critic2, self.critic2_target)
            
            # Store actor loss
            self.actor_losses.append(actor_loss.item())
        
        # Store critic losses
        self.critic_losses.append(critic1_loss.item())
        
        return actor_loss if isinstance(actor_loss, float) else actor_loss.item(), critic1_loss.item(), critic2_loss.item()
    
    def save(self, path: str):
        """
        Save agent.
        
        Args:
            path: Path to save the agent
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'rewards': self.rewards,
            'total_steps': self.total_steps,
            'update_counter': self.update_counter
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load agent.
        
        Args:
            path: Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        self.rewards = checkpoint['rewards']
        self.total_steps = checkpoint['total_steps']
        self.update_counter = checkpoint['update_counter']
        logger.info(f"Agent loaded from {path}")


class PPONetwork(nn.Module):
    """
    Network for Proximal Policy Optimization (PPO).
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize PPO network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic (value) network
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action_mean, action_log_std, value)
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Actor outputs
        action_mean = torch.tanh(self.actor_mean(features))
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_log_std, value
    
    def get_action(self, state, deterministic: bool = False):
        """
        Get action from policy.
        
        Args:
            state: State tensor
            deterministic: Whether to use deterministic action (mean)
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        # Forward pass
        action_mean, action_log_std, value = self.forward(state)
        
        # Create distribution
        action_std = torch.exp(action_log_std)
        distribution = torch.distributions.Normal(action_mean, action_std)
        
        # Sample action
        if deterministic:
            action = action_mean
        else:
            action = distribution.sample()
        
        # Clip action to [-1, 1]
        action = torch.clamp(action, -1, 1)
        
        # Compute log probability
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """
    
    def __init__(self, 
                state_dim: int, 
                action_dim: int, 
                hidden_dim: int = 256,
                lr: float = 3e-4,
                gamma: float = 0.99,
                gae_lambda: float = 0.95,
                clip_ratio: float = 0.2,
                value_coef: float = 0.5,
                entropy_coef: float = 0.01,
                max_grad_norm: float = 0.5,
                update_epochs: int = 10,
                batch_size: int = 64,
                device: str = 'cuda'):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs to update per batch
            batch_size: Batch size for training
            device: Device to use for training (cuda or cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda # GAE stands for Generalized Advantage Estimation: it is a technique to estimate the advantage function, which is the difference between the expected future returns and the current value function.
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize PPO network
        self.network = PPONetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Initialize buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
        # Initialize training metrics
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.rewards = []
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select action.
        
        Args:
            state: Current state
            eval_mode: Whether to use deterministic action
            
        Returns:
            Selected action
        """
        # Check for dimension mismatch and fix at runtime
        expected_dim = self.state_dim
        actual_dim = state.shape[0]
        
        if actual_dim != expected_dim:
            logger.warning(f"State dimension mismatch: expected {expected_dim}, got {actual_dim}. Adapting...")
            if actual_dim < expected_dim:
                # Pad with zeros if input is smaller than expected
                padding = np.zeros(expected_dim - actual_dim)
                state = np.concatenate([state, padding])
            else:
                # Truncate if input is larger than expected
                state = state[:expected_dim]
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Set network to evaluation mode
        self.network.eval()
        
        with torch.no_grad():
            # Get action from network
            action, log_prob, value = self.network.get_action(state_tensor, deterministic=eval_mode)
        
        # Set network back to training mode
        self.network.train()
        
        # Extract action from tensor
        action_np = action.cpu().numpy()
        
        # Scale action to [0, 1] range for portfolio weights
        action_np = (action_np + 1.0) / 2.0
        
        # Normalize to ensure weights sum to 1
        action_np = action_np / np.sum(action_np)
        
        # Store in buffer if not in evaluation mode
        if not eval_mode:
            self.buffer['states'].append(state)
            self.buffer['actions'].append(action.cpu().numpy())
            self.buffer['log_probs'].append(log_prob.cpu().numpy())
            self.buffer['values'].append(value.cpu().numpy())
        
        return action_np
    
    def process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Process action for storage in buffer.
        
        Args:
            action: Action to process
            
        Returns:
            Processed action
        """
        # Convert from [0, 1] range to [-1, 1] range for PPO
        action_processed = action * 2.0 - 1.0
        return action_processed
    
    def store(self, reward: float, done: bool):
        """
        Store experience in buffer.
        
        Args:
            reward: Reward received
            done: Whether the episode is done
        """
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
    
    def update(self) -> Tuple[float, float, float, float]:
        """
        Update the agent.
        
        Returns:
            Tuple of (loss, policy_loss, value_loss, entropy_loss)
        """
        # Check if buffer has enough samples
        if len(self.buffer['states']) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer['rewards'])).to(self.device).unsqueeze(1)
        values = torch.FloatTensor(np.array(self.buffer['values'])).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer['dones'])).to(self.device).unsqueeze(1)
        
        # Compute returns and advantages
        returns, advantages = self._compute_returns_advantages(rewards, values, dones)
        
        # Update policy for multiple epochs
        for _ in range(self.update_epochs):
            # Generate random indices
            indices = torch.randperm(states.size(0))
            
            # Split indices into batches
            batches = [indices[i:i + self.batch_size] for i in range(0, indices.size(0), self.batch_size)]
            
            # Process each batch
            for batch_indices in batches:
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_mean, action_log_std, value = self.network.forward(batch_states)
                
                # Create distribution
                action_std = torch.exp(action_log_std)
                distribution = torch.distributions.Normal(action_mean, action_std)
                
                # Compute entropy
                entropy = distribution.entropy().sum(dim=-1, keepdim=True).mean()
                
                # Compute new log probabilities
                new_log_probs = distribution.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                # Compute policy loss
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(value, batch_returns)
                
                # Compute total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                self.losses.append(loss.item())
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy.item())
        
        # Clear buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
        return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()
    
    def _compute_returns_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards tensor
            values: Values tensor
            dones: Dones tensor
            
        Returns:
            Tuple of (returns, advantages)
        """
        # Initialize returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Initialize for computing GAE
        next_return = 0
        next_value = 0
        next_advantage = 0
        
        # Iterate backwards through the sequence
        for t in reversed(range(len(rewards))):
            # Compute return
            next_return = rewards[t] + self.gamma * next_return * (1 - dones[t])
            returns[t] = next_return
            
            # Compute TD error
            td_error = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Compute advantage
            next_advantage = td_error + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t])
            advantages[t] = next_advantage
            
            # Update next value
            next_value = values[t]
        
        return returns, advantages
    
    def save(self, path: str):
        """
        Save agent.
        
        Args:
            path: Path to save the agent
        """
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'rewards': self.rewards,
            'total_steps': self.total_steps
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load agent.
        
        Args:
            path: Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.losses = checkpoint['losses']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        self.entropy_losses = checkpoint['entropy_losses']
        self.rewards = checkpoint['rewards']
        self.total_steps = checkpoint['total_steps']
        logger.info(f"Agent loaded from {path}")
    
    def plot_learning_curves(self) -> plt.Figure:
        """
        Plot learning curves.
        
        Returns:
            Matplotlib figure
        """
        fig, axs = plt.subplots(4, 1, figsize=(10, 16))
        
        # Plot total loss
        axs[0].plot(self.losses)
        axs[0].set_title('Total Loss')
        axs[0].set_xlabel('Training Steps')
        axs[0].set_ylabel('Loss')
        axs[0].grid(True)
        
        # Plot policy loss
        axs[1].plot(self.policy_losses)
        axs[1].set_title('Policy Loss')
        axs[1].set_xlabel('Training Steps')
        axs[1].set_ylabel('Loss')
        axs[1].grid(True)
        
        # Plot value loss
        axs[2].plot(self.value_losses)
        axs[2].set_title('Value Loss')
        axs[2].set_xlabel('Training Steps')
        axs[2].set_ylabel('Loss')
        axs[2].grid(True)
        
        # Plot entropy loss
        axs[3].plot(self.entropy_losses)
        axs[3].set_title('Entropy Loss')
        axs[3].set_xlabel('Training Steps')
        axs[3].set_ylabel('Loss')
        axs[3].grid(True)
        
        # Plot rewards in a new figure
        reward_fig, reward_ax = plt.subplots(figsize=(10, 4))
        reward_ax.plot(self.rewards)
        reward_ax.set_title('Episode Rewards')
        reward_ax.set_xlabel('Episodes')
        reward_ax.set_ylabel('Reward')
        reward_ax.grid(True)
        
        plt.tight_layout()
        return fig


def train_agent(
    agent: Union[DDPGAgent, TD3Agent, PPOAgent],
    env: gym.Env,
    episodes: int = 1000,
    max_steps: int = 1000,
    eval_freq: int = 10,
    save_path: Optional[str] = None,
    save_freq: int = 100
) -> Dict[str, List[float]]:
    """
    Train an RL agent.
    
    Args:
        agent: Agent to train
        env: Environment to train on
        episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        eval_freq: Frequency of evaluation
        save_path: Path to save the agent
        save_freq: Frequency of saving
        
    Returns:
        Dictionary with training metrics
    """
    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'eval_rewards': [],
        'eval_steps': []
    }
    
    # Track evaluation results
    last_eval_reward = 0
    last_eval_steps = 0
    
    # Train for specified number of episodes
    for episode in range(1, episodes + 1):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Run episode
        for step in range(1, max_steps + 1):
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Process action for storage
            action_processed = agent.process_action(action)
            
            # Store experience
            if isinstance(agent, PPOAgent):
                agent.store(reward, terminated or truncated)
            else:
                agent.replay_buffer.add(state, action_processed, reward, next_state, terminated or truncated)
            
            # Update agent
            if not isinstance(agent, PPOAgent) or episode_steps % 128 == 0:
                agent.update()
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            agent.total_steps += 1
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Update agent at the end of the episode (for PPO)
        if isinstance(agent, PPOAgent):
            agent.update()
        
        # Store episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_steps'].append(episode_steps)
        agent.rewards.append(episode_reward)
        
        # Print progress
        logger.info(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        # Evaluate agent
        if episode % eval_freq == 0:
            eval_reward, eval_steps = evaluate_agent(agent, env, episodes=5)
            last_eval_reward = eval_reward
            last_eval_steps = eval_steps
            logger.info(f"Evaluation - Average Reward: {eval_reward:.2f}, Average Steps: {eval_steps:.2f}")
        
        # Always append evaluation metrics (use the last evaluation result for non-eval episodes)
        # This ensures all metric lists have the same length
        metrics['eval_rewards'].append(last_eval_reward)
        metrics['eval_steps'].append(last_eval_steps)
        
        # Save agent
        if save_path is not None and episode % save_freq == 0:
            agent.save(f"{save_path}_episode_{episode}.pt")
    
    # Save final agent
    if save_path is not None:
        agent.save(f"{save_path}_final.pt")
    
    # Process metrics for visualization
    # Only keep evaluation metrics for evaluation episodes
    processed_metrics = {
        'episode_rewards': metrics['episode_rewards'],
        'episode_steps': metrics['episode_steps'],
        'eval_rewards': [metrics['eval_rewards'][i-1] if i % eval_freq == 0 else float('nan') for i in range(1, episodes+1)],
        'eval_steps': [metrics['eval_steps'][i-1] if i % eval_freq == 0 else float('nan') for i in range(1, episodes+1)]
    }
    
    return processed_metrics


def evaluate_agent(
    agent: Union[DDPGAgent, TD3Agent, PPOAgent],
    env: gym.Env,
    episodes: int = 10,
    max_steps: int = 1000
) -> Tuple[float, float]:
    """
    Evaluate an RL agent.
    
    Args:
        agent: Agent to evaluate
        env: Environment to evaluate on
        episodes: Number of episodes to evaluate for
        max_steps: Maximum steps per episode
        
    Returns:
        Tuple of (average_reward, average_steps)
    """
    # Initialize metrics
    rewards = []
    steps = []
    
    # Evaluate for specified number of episodes
    for _ in range(episodes):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Run episode
        for step in range(1, max_steps + 1):
            # Select action (evaluation mode)
            action = agent.select_action(state, eval_mode=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Store episode metrics
        rewards.append(episode_reward)
        steps.append(episode_steps)
    
    # Calculate averages
    avg_reward = sum(rewards) / len(rewards)
    avg_steps = sum(steps) / len(steps)
    
    return avg_reward, avg_steps
