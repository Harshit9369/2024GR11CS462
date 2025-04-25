import numpy as np

class CellularEnvironment:
    def __init__(self, num_gnbs=5, num_ues=20, grid_size=1000, episode_length=200):
        self.num_gnbs = num_gnbs
        self.num_ues = num_ues
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.current_step = 0
        
        self.gnbs = []
        for i in range(num_gnbs):
            self.gnbs.append({
                'id': i,
                'position': np.random.rand(2) * grid_size,
                'tx_power': 30 + np.random.rand() * 10,
                'frequency': 3500 + (i % 3) * 100,
                'capacity': 100,
                'load': 0,
                'neighbors': []
            })
        
        self._establish_gnb_neighbors()
        
        self.ues = []
        for i in range(num_ues):
            velocity = np.random.rand(2) * 20 - 10
            self.ues.append({
                'id': i,
                'position': np.random.rand(2) * grid_size,
                'velocity': velocity,
                'connected_gnb': None,
                'rsrp_measurements': np.zeros(num_gnbs),
                'throughput': 0,
                'handover_history': [],
                'ping_pong': 0,
                'last_handover_time': -100
            })
        
        self._initial_connections()
    
    def _establish_gnb_neighbors(self):
        for i in range(self.num_gnbs):
            for j in range(self.num_gnbs):
                if i != j:
                    dist = np.linalg.norm(self.gnbs[i]['position'] - self.gnbs[j]['position'])
                    if dist < self.grid_size * 0.4:
                        self.gnbs[i]['neighbors'].append(j)
    
    def _calculate_path_loss(self, ue_pos, gnb_pos, gnb_power):
        distance = np.linalg.norm(ue_pos - gnb_pos)
        distance_km = max(0.001, distance / 1000)
        path_loss = 128.1 + 37.6 * np.log10(distance_km)
        shadowing = np.random.normal(0, 8)
        rsrp = gnb_power - path_loss + shadowing
        return rsrp
    
    def _initial_connections(self):
        for ue_idx, ue in enumerate(self.ues):
            for gnb_idx, gnb in enumerate(self.gnbs):
                rsrp = self._calculate_path_loss(ue['position'], gnb['position'], gnb['tx_power'])
                self.ues[ue_idx]['rsrp_measurements'][gnb_idx] = rsrp
            
            best_gnb = np.argmax(ue['rsrp_measurements'])
            self.ues[ue_idx]['connected_gnb'] = best_gnb
            self.gnbs[best_gnb]['load'] += 1
    
    def _update_ue_positions(self):
        for ue_idx, ue in enumerate(self.ues):
            new_position = ue['position'] + ue['velocity']
            new_position = np.clip(new_position, 0, self.grid_size)
            velocity_change = (np.random.rand(2) - 0.5) * 0.5
            new_velocity = ue['velocity'] + velocity_change
            speed = np.linalg.norm(new_velocity)
            if speed > 10:
                new_velocity = new_velocity * (10 / speed)
            self.ues[ue_idx]['position'] = new_position
            self.ues[ue_idx]['velocity'] = new_velocity
    
    def _update_measurements(self):
        for ue_idx, ue in enumerate(self.ues):
            for gnb_idx, gnb in enumerate(self.gnbs):
                rsrp = self._calculate_path_loss(ue['position'], gnb['position'], gnb['tx_power'])
                self.ues[ue_idx]['rsrp_measurements'][gnb_idx] = rsrp
    
    def _calculate_throughput(self, ue_idx):
        ue = self.ues[ue_idx]
        gnb_idx = ue['connected_gnb']
        gnb = self.gnbs[gnb_idx]
        signal = 10 ** (ue['rsrp_measurements'][gnb_idx] / 10)
        interference = 0
        for i, other_gnb in enumerate(self.gnbs):
            if i != gnb_idx and other_gnb['frequency'] == gnb['frequency']:
                interference += 10 ** (ue['rsrp_measurements'][i] / 10)
        noise_floor = 10 ** (-104 / 10)
        sinr = signal / (interference + noise_floor)
        sinr_db = 10 * np.log10(sinr)
        bandwidth = 20
        spectral_efficiency = np.log2(1 + sinr)
        load_factor = max(0.1, 1 - (gnb['load'] / gnb['capacity']))
        throughput = bandwidth * spectral_efficiency * load_factor
        return throughput
    
    def _check_ping_pong(self, ue_idx, target_gnb):
        ue = self.ues[ue_idx]
        history = ue['handover_history']
        if len(history) >= 1 and history[-1] == target_gnb and self.current_step - ue['last_handover_time'] < 5:
            return True
        return False
    
    def _perform_xn_handover(self, ue_idx, target_gnb):
        ue = self.ues[ue_idx]
        source_gnb = ue['connected_gnb']
        if target_gnb not in self.gnbs[source_gnb]['neighbors']:
            return False, "No Xn interface between gNBs"
        if self.gnbs[target_gnb]['load'] >= self.gnbs[target_gnb]['capacity']:
            return False, "Target gNB at capacity"
        self.gnbs[source_gnb]['load'] -= 1
        self.gnbs[target_gnb]['load'] += 1
        self.ues[ue_idx]['connected_gnb'] = target_gnb
        self.ues[ue_idx]['handover_history'].append(target_gnb)
        ping_pong = self._check_ping_pong(ue_idx, target_gnb)
        if ping_pong:
            self.ues[ue_idx]['ping_pong'] += 1
        self.ues[ue_idx]['last_handover_time'] = self.current_step
        new_throughput = self._calculate_throughput(ue_idx)
        self.ues[ue_idx]['throughput'] = new_throughput
        return True, "Handover successful"
    
    def step(self):
        self.current_step += 1
        self._update_ue_positions()
        self._update_measurements()
        for ue_idx in range(self.num_ues):
            self.ues[ue_idx]['throughput'] = self._calculate_throughput(ue_idx)
        done = self.current_step >= self.episode_length
        return done
    
    def reset(self):
        self.current_step = 0
        for gnb in self.gnbs:
            gnb['load'] = 0
        for ue in self.ues:
            ue['position'] = np.random.rand(2) * self.grid_size
            ue['velocity'] = np.random.rand(2) * 20 - 10
            ue['connected_gnb'] = None
            ue['throughput'] = 0
            ue['handover_history'] = []
            ue['ping_pong'] = 0
            ue['last_handover_time'] = -100
        self._initial_connections()
        return self._get_state()
    
    def _get_state(self):
        states = []
        for ue_idx, ue in enumerate(self.ues):
            state = {
                'ue_id': ue_idx,
                'rsrp': ue['rsrp_measurements'].copy(),
                'velocity': np.linalg.norm(ue['velocity']),
                'gnb_loads': np.array([gnb['load'] / gnb['capacity'] for gnb in self.gnbs]),
                'time_since_ho': self.current_step - ue['last_handover_time']
            }
            states.append(state)
        return states
    
    def get_interference_level(self, ue_idx, target_gnb):
        ue = self.ues[ue_idx]
        interference = 0
        target_freq = self.gnbs[target_gnb]['frequency']
        for i, gnb in enumerate(self.gnbs):
            if i != target_gnb and gnb['frequency'] == target_freq:
                interference += 10 ** (ue['rsrp_measurements'][i] / 10)
        interference = 10 * np.log10(interference + 1e-10)
        normalized_interference = (interference + 120) / 70
        normalized_interference = np.clip(normalized_interference, 0, 1)
        return normalized_interference
