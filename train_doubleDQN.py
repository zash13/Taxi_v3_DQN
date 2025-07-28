import gymnasium as gym
import numpy as np
from DQN.DQN_Agent import (
    EpsilonPolicyType,
    EpsilonPolicy,
    DoubleDQNAgent,
    RewardPolicyType,
    UpdateTargetNetworkType,
)

env = gym.make("Taxi-v3")

state_size = env.observation_space.n  # 500
action_size = env.action_space.n  # 6

epsilon_min = 0.01
epsilon_decay = 0.995
ep_policy = EpsilonPolicy(
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    policy=EpsilonPolicyType.DECAY,
)

num_episodes = 1000
agent = DoubleDQNAgent(
    action_size=action_size,
    state_size=state_size,
    learning_rate=0.001,
    buffer_size=2000,
    batch_size=32,
    gamma=0.99,
    max_episodes=num_episodes,
    epsilon=1.0,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    epsilon_policy=ep_policy,
    reward_policy=RewardPolicyType.ERM,
    prefer_lower_heuristic=False,
    progress_bonus=0.05,
    exploration_bonus=0.1,
    update_target_network_method=UpdateTargetNetworkType.HARD,
    update_factor=0.005,
    target_update_frequency=10,
)

rewards = []
max_steps = 200
counter = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.eye(state_size)[state]
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        print(f"Episode {episode}, Step {step}, Counter: {counter}")
        counter += 1
        action = agent.select_action(np.array([state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.eye(state_size)[next_state]
        done = terminated or truncated

        print(
            f"Action: {action}, Reward: {reward}, Done: {done}, Terminated: {terminated}, Truncated: {truncated}"
        )

        agent.buffer_helper.store_experience(
            current_state=state,
            next_state=next_state,
            imm_reward=reward,
            action=action,
            done=done,
            heuristic=None,
        )

        loss = agent.train(episode)
        print(f"Loss: {loss if loss is not None else 'None'}")

        state = next_state
        total_reward += reward
        step += 1

    rewards.append(total_reward)
    print(
        f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, "
        "Epsilon: {agent.epsilon:.3f}, "
        f"Buffer Size: {len(agent.buffer_helper.memory_buffer)}, Loss: {loss if loss is not None else 'None'}"
    )

env.close()

import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress on Taxi-v3 with Double DQN")
plt.show()
