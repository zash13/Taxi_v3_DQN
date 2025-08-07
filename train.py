import gymnasium as gym
import numpy as np
import time
from DQN.DQN_Agent import (
    AgentFactory,
    AgentType,
    EpsilonPolicyType,
    EpsilonPolicy,
    UpdateTargetNetworkType,
    RewardPolicyType,
)

import matplotlib.pyplot as plt


def main():
    env = gym.make("Taxi-v3")

    state_size = env.observation_space.n  # Total number of discrete states
    action_size = env.action_space.n  # Number of possible actions

    epsilon_min = 0.1
    epsilon_decay = 0.995
    ep_policy = EpsilonPolicy(
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        policy=EpsilonPolicyType.DECAY,
    )

    num_episodes = 1000
    max_steps = 200
    render = False

    agent = AgentFactory.create_agent(
        AgentType.DQN,  # or DOUBLE_DQN if supported
        action_size=action_size,
        state_size=state_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=32,
        buffer_size=2000,
        max_episodes=num_episodes,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        epsilon_policy=ep_policy,
        reward_policy=RewardPolicyType.ERM,
        progress_bonus=0.05,
        exploration_bonus=0.1,
    )

    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_onehot = np.eye(state_size)[state]  # One-hot encode state
        total_reward = 0
        step = 0
        done = False

        while not done and step < max_steps:
            action = agent.select_action(np.array([state_onehot]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_onehot = np.eye(state_size)[next_state]
            done = terminated or truncated

            agent.store_experience(
                state_onehot, next_state_onehot, reward, action, done, huristic=None
            )
            loss = agent.train(episode)

            state_onehot = next_state_onehot
            total_reward += reward
            step += 1

            if render:
                env.render()
                time.sleep(0.05)

        rewards.append(total_reward)
        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, "
            f"Epsilon: {agent.get_epsilon():.3f}, Loss: {loss}"
        )

    env.close()

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress on Taxi-v3")
    plt.show()


if __name__ == "__main__":
    main()
