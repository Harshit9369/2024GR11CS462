import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class HandoverRL:
    def __init__(self, env, use_dqn=True):
        self.env = env
        self.use_dqn = use_dqn
        self.num_gnbs = env.num_gnbs
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        if not use_dqn:
            self.q_tables = [np.zeros(self.num_gnbs) for _ in range(env.num_ues)]
        else:
            self.memory = deque(maxlen=2000)
            self.batch_size = 32
            self.models = [self._build_model() for _ in range(env.num_ues)]
    
    def _build_model(self):
        input_dim = self.num_gnbs * 2 + 2
        model = Sequential()
        model.add(Dense(24, input_dim=input_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.num_gnbs, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def _state_to_input(self, state):
        inputs = np.concatenate([
            state['rsrp'],
            [state['velocity']],
            state['gnb_loads'],
            [state['time_since_ho']]
        ])
        return inputs
    
    def remember(self, ue_idx, state, action, reward, next_state, done):
        if self.use_dqn:
            state_input = self._state_to_input(state)
            next_state_input = self._state_to_input(next_state)
            self.memory.append((state_input, action, reward, next_state_input, done))
    
    def act(self, ue_idx, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_gnbs)
        if not self.use_dqn:
            return np.argmax(self.q_tables[ue_idx])
        else:
            state_input = self._state_to_input(state)
            state_input = np.reshape(state_input, [1, -1])
            q_values = self.models[ue_idx].predict(state_input, verbose=0)[0]
            return np.argmax(q_values)
    
    def train(self, ue_idx, state, action, reward, next_state, done):
        if not self.use_dqn:
            current_q = self.q_tables[ue_idx][action]
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.q_tables[ue_idx])
            self.q_tables[ue_idx][action] += self.learning_rate * (target_q - current_q)
        else:
            self.remember(ue_idx, state, action, reward, next_state, done)
            if len(self.memory) >= self.batch_size:
                self._replay(ue_idx)
    
    def _replay(self, ue_idx):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, -1])
            next_state = np.reshape(next_state, [1, -1])
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.models[ue_idx].predict(next_state, verbose=0)[0])
            target_f = self.models[ue_idx].predict(state, verbose=0)
            target_f[0][action] = target
            self.models[ue_idx].fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, ue_idx, target_gnb):
        ue = self.env.ues[ue_idx]
        current_gnb = ue['connected_gnb']
        if current_gnb == target_gnb:
            return 0
        old_gnb = current_gnb
        old_throughput = ue['throughput']
        success, _ = self.env._perform_xn_handover(ue_idx, target_gnb)
        if not success:
            self.env._perform_xn_handover(ue_idx, old_gnb)
            return -10
        new_throughput = ue['throughput']
        r_p = np.log(1 + new_throughput)
        T = 10
        recent_ho_count = sum(1 for t in [ue['last_handover_time']] 
                              if t > self.env.current_step - T)
        r_s = -0.5 * recent_ho_count
        interference_level = self.env.get_interference_level(ue_idx, target_gnb)
        r_i = -0.3 * interference_level
        total_reward = r_p + r_s + r_i
        self.env._perform_xn_handover(ue_idx, old_gnb)
        return total_reward
