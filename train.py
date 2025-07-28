import gymnasium as gym
import numpy as np
from DQN.DQN_Agent import EpsilonPolicyType, EpsilonPolicy, DQNAgent, RewardPolicyType

env = gym.make("Taxi-v3")

state_size = env.observation_space.n
action_size = env.action_space.n

epsilon_min = 0.1
epsilon_decay = 0.995
ep_policy = EpsilonPolicy(
    epsilon_min,
    epsilon_decay,
    policy=EpsilonPolicyType.DECAY,
)
num_episodes = 1000
rewards = []
max_steps = 200
agent = DQNAgent(
    action_size=action_size,
    state_size=state_size,
    learning_rate=0.001,
    buffer_size=2000,
    batch_size=32,
    gamma=0.99,
    max_episodes=num_episodes,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    epsilon_policy=ep_policy,
    reward_policy=RewardPolicyType.ERM,
    progress_bonus=0.05,
    exploration_bonus=0.1,
)
for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.eye(state_size)[state]
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        action = agent.select_action(np.array([state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.eye(state_size)[next_state]
        done = terminated or truncated

        agent.buffer_helper.store_experience(
            current_state=state,
            next_state=next_state,
            imm_reward=reward,
            action=action,
            done=done,
            heuristic=None,
        )

        loss = agent.train(episode)

        state = next_state
        total_reward += reward
        step += 1

    rewards.append(total_reward)
    print(
        f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f} , loss: {loss}"
    )

env.close()

import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress on Taxi-v3")
plt.show()
