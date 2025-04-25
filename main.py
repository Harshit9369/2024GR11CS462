import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from environment import CellularEnvironment
from rl_algo import HandoverRL

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def train_rl_agent(env, agent, episodes=100):
    rewards_history = []
    
    for episode in range(episodes):
        states = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            episode_rewards = []
            
            for ue_idx, state in enumerate(states):
                current_gnb = env.ues[ue_idx]['connected_gnb']
                target_gnb = agent.act(ue_idx, state)
                
                if target_gnb == current_gnb:
                    continue
                
                reward = agent.calculate_reward(ue_idx, target_gnb)
                
                if reward > 0:
                    success, _ = env._perform_xn_handover(ue_idx, target_gnb)
                    if success:
                        episode_rewards.append(reward)
                
                next_done = env.step()
                next_states = env._get_state()
                agent.train(ue_idx, state, target_gnb, reward, next_states[ue_idx], next_done)
                done = next_done
            
            if episode_rewards:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                total_reward += avg_reward
        
        rewards_history.append(total_reward)
        
        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    return rewards_history

def traditional_handover(env, episodes=100):
    rewards_history = []
    
    for episode in range(episodes):
        states = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            episode_rewards = []
            
            for ue_idx, state in enumerate(states):
                ue = env.ues[ue_idx]
                current_gnb = ue['connected_gnb']
                current_rsrp = ue['rsrp_measurements'][current_gnb]
                hysteresis = 3
                rsrp_measurements = ue['rsrp_measurements']
                best_gnb = np.argmax(rsrp_measurements)
                best_rsrp = rsrp_measurements[best_gnb]
                
                if best_gnb != current_gnb and best_rsrp > current_rsrp + hysteresis:
                    success, _ = env._perform_xn_handover(ue_idx, best_gnb)
                    
                    if success:
                        reward = np.log(1 + ue['throughput'])
                        episode_rewards.append(reward)
            
            done = env.step()
            
            if episode_rewards:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                total_reward += avg_reward
        
        rewards_history.append(total_reward)
        
        print(f"Traditional Episode: {episode}, Total Reward: {total_reward:.2f}")
    
    return rewards_history

def create_handover_dataset(env, num_samples=1000):
    data = []
    env.reset()
    
    for _ in range(num_samples):
        env.step()
        
        for ue_idx, ue in enumerate(env.ues):
            current_gnb = ue['connected_gnb']
            rsrp_values = ue['rsrp_measurements']
            best_gnb = np.argmax(rsrp_values)
            current_throughput = ue['throughput']
            env._perform_xn_handover(ue_idx, best_gnb)
            best_throughput = ue['throughput']
            env._perform_xn_handover(ue_idx, current_gnb)
            
            sample = {
                'ue_id': ue_idx,
                'current_gnb': current_gnb,
                'best_rsrp_gnb': best_gnb,
                'rsrp_values': rsrp_values.tolist(),
                'gnb_loads': [gnb['load'] / gnb['capacity'] for gnb in env.gnbs],
                'velocity': np.linalg.norm(ue['velocity']),
                'current_throughput': current_throughput,
                'best_gnb_throughput': best_throughput,
                'time_since_last_ho': env.current_step - ue['last_handover_time']
            }
            data.append(sample)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    env = CellularEnvironment(num_gnbs=5, num_ues=20, grid_size=1000, episode_length=100)
    print("Creating handover dataset...")
    handover_data = create_handover_dataset(env, num_samples=500)
    print(f"Dataset created with {len(handover_data)} samples")
    agent = HandoverRL(env, use_dqn=True)
    print("\nTraining RL agent...")
    rl_rewards = train_rl_agent(env, agent, episodes=50)
    print("\nEvaluating traditional handover algorithm...")
    traditional_rewards = traditional_handover(env, episodes=50)
    plt.figure(figsize=(10, 6))
    plt.plot(rl_rewards, label='RL-Based Handover')
    plt.plot(traditional_rewards, label='Traditional A3 Handover')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('RL vs Traditional Handover Performance')
    plt.legend()
    plt.grid(True)
    handover_data.to_csv('handover_dataset.csv', index=False)
    print("Dataset saved to handover_dataset.csv")
    print("\nTraining completed!")
    print(f"Final RL average reward: {np.mean(rl_rewards[-10:]):.2f}")
    print(f"Final Traditional average reward: {np.mean(traditional_rewards[-10:]):.2f}")
