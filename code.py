# final_complete_ieee_clean.py
# COMPLETE IEEE-READY SIMULATION WITH CLEAN OUTPUT STRUCTURE

import csv
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import json
from datetime import datetime
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# =============================================
# SIMULATION CONFIGURATION
# =============================================

class SimulationConfig:
    """Centralized configuration for all simulation parameters"""
    
    # Time settings
    DT = 5  # seconds per time step
    SIM_MINUTES = 180  # 3-hour simulation
    SIM_SECONDS = SIM_MINUTES * 60
    
    # Fleet settings
    NUM_BUSES = 12
    SEED = 42
    
    # RL Training
    TRAINING_EPISODES = 200
    TRAINING_EPISODE_SECONDS = 3600  # 60 minutes per training episode
    
    # Bus specifications (BYD K9 Shenzhen)
    BATTERY_KWH = 350.0
    CONSUMPTION_KWH_PER_KM = 1.65
    CRUISE_SPEED_KMH = 32.0
    
    # Charging parameters
    INIT_SOC_RANGE = (0.40, 0.60)
    DOCK_SOC_THRESHOLD = 0.40
    CHARGE_TARGET_SOC = 0.80  # Charging stops at this SOC (used consistently)
    DOCK_PROXIMITY_KM = 1.0
    
    # Route - Shenzhen Line 303 (42.1 km one-way, 84.2 km round trip)
    ROUTE_WAYPOINTS = [
        (0.0,0.0),(0.8,0.2),(1.6,0.5),(2.3,0.9),(3.0,1.4),(3.7,2.0),(4.3,2.7),(4.8,3.5),(5.3,4.3),(5.7,5.2),
        (6.0,6.1),(6.3,7.0),(6.5,8.0),(6.7,9.0),(6.8,10.0),(6.9,11.0),(7.0,12.0),(7.0,13.0),(7.0,14.0),(7.0,15.0),
        (7.0,16.0),(7.0,17.0),(7.0,18.0),(7.0,19.0),(7.0,20.0),(7.0,21.0),(7.0,22.0),(7.0,22.5),
        (7.2,23.0),(7.5,24.0),(8.0,25.0),(8.5,26.0),(9.0,27.0),(9.5,28.0),(10.0,29.0),(10.5,30.0),(11.0,31.0),
        (11.5,32.0),(12.0,33.0),(12.5,34.0),(13.0,35.0),(13.5,36.0),(14.0,37.0),(14.5,38.0),(15.0,39.0),(15.5,40.0),
        (16.0,41.0),(16.5,42.0),(17.0,43.0),(17.5,44.0),(18.0,45.0),(18.5,46.0),(19.0,47.0),(19.5,48.0),(20.0,49.0),
        (20.5,50.0),(21.0,51.0),(21.5,52.0),(22.0,53.0),(22.0,53.5),
        # Return trip
        (21.5,53.0),(21.0,52.0),(20.5,51.0),(20.0,50.0),(19.5,49.0),(19.0,48.0),(18.5,47.0),(18.0,46.0),(17.5,45.0),
        (17.0,44.0),(16.5,43.0),(16.0,42.0),(15.5,41.0),(15.0,40.0),(14.5,39.0),(14.0,38.0),(13.5,37.0),(13.0,36.0),
        (12.5,35.0),(12.0,34.0),(11.5,33.0),(11.0,32.0),(10.5,31.0),(10.0,30.0),(9.5,29.0),(9.0,28.0),(8.5,27.0),
        (8.0,26.0),(7.5,25.0),(7.0,24.0),(6.5,23.0),(6.0,22.0),(5.5,21.0),(5.0,20.0),(4.5,19.0),(4.0,18.0),(3.5,17.0),
        (3.0,16.0),(2.5,15.0),(2.0,14.0),(1.5,13.0),(1.0,12.0),(0.5,11.0),(0.0,10.0),(-0.5,9.0),(-1.0,8.0),
        (-1.5,7.0),(-2.0,6.0),(-2.5,5.0),(-3.0,4.0),(-3.5,3.0),(-4.0,2.0),(-4.5,1.0),(-5.0,0.0)
    ]
    
    # Charging infrastructure
    CHARGER_LOCATIONS = [
        (2.0, 1.0),     # Near Bao'an terminal
        (7.0, 12.0),    # Mid-route hub
        (7.0, 22.0),    # City center
        (14.0, 37.0),   # Mid-return
        (16.0, 41.0),   # Near Luohu terminal
    ]
    PORTS_PER_CHARGER = 2
    CHARGING_POWER_KW = 120.0
    
    # Strategy comparison
    STRATEGIES = ["baseline", "heuristic", "random", "oracle", "rl"]
    
    # RL hyperparameters
    RL_STATE_SOC_BUCKETS = 6
    RL_STATE_DIST_BUCKETS = 5
    RL_ALPHA = 0.1
    RL_GAMMA = 0.95
    RL_EPS_START = 1.0
    RL_EPS_DECAY = 0.995
    RL_MIN_EPS = 0.05
    
    # Visualization
    EMA_ALPHA = 0.3

config = SimulationConfig()

# =============================================
# OUTPUT MANAGER - CLEAN ORGANIZATION
# =============================================

class OutputManager:
    """Manages all output files with organized structure"""
    
    def __init__(self, base_name="shenzhen_bus_simulation"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f"{base_name}_{timestamp}"
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Create organized folder structure"""
        # Main directories
        self.dirs = {
            'root': self.base_dir,
            'data': os.path.join(self.base_dir, 'data'),
            'plots': os.path.join(self.base_dir, 'plots'),
            'animations': os.path.join(self.base_dir, 'animations'),
            'paper': os.path.join(self.base_dir, 'paper_materials'),
            'logs': os.path.join(self.base_dir, 'logs'),
            'tables': os.path.join(self.base_dir, 'tables')
        }
        
        # Strategy-specific directories
        for strategy in config.STRATEGIES:
            self.dirs[f'data_{strategy}'] = os.path.join(self.dirs['data'], strategy)
            self.dirs[f'plots_{strategy}'] = os.path.join(self.dirs['plots'], strategy)
            self.dirs[f'animations_{strategy}'] = os.path.join(self.dirs['animations'], strategy)
        
        # Create all directories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Create README
        self._create_readme()
    
    def _create_readme(self):
        """Create README file explaining the structure"""
        readme_content = f"""# Shenzhen Electric Bus Simulation - {datetime.now().strftime('%Y-%m-%d')}

## DIRECTORY STRUCTURE:
/
├── data/                    # All simulation data
│   ├── baseline/           # Baseline strategy data
│   ├── heuristic/          # Heuristic strategy data  
│   ├── random/            # Random strategy data
│   ├── oracle/            # Oracle (optimal) strategy data
│   └── rl/                # Reinforcement learning strategy data
│
├── plots/                  # All visualization plots
│   ├── baseline/          # Baseline strategy plots
│   ├── heuristic/         # Heuristic strategy plots
│   ├── random/           # Random strategy plots
│   ├── oracle/           # Oracle strategy plots
│   ├── rl/               # RL strategy plots
│   ├── comparisons/      # Strategy comparison plots
│   └── training/         # RL training progress plots
│
├── animations/            # Simulation animations (GIF/MP4)
│   ├── baseline/         # Baseline animations
│   ├── heuristic/        # Heuristic animations
│   ├── random/          # Random animations
│   ├── oracle/          # Oracle animations
│   └── rl/              # RL animations
│
├── paper_materials/       # IEEE paper materials
│   ├── latex/           # LaTeX source files
│   ├── figures/         # Paper figures
│   └── tables/          # Result tables
│
├── logs/                 # Simulation logs
└── tables/              # Statistical result tables

## SIMULATION PARAMETERS:
- Buses: {config.NUM_BUSES}
- Charging stations: {len(config.CHARGER_LOCATIONS)}
- Ports per station: {config.PORTS_PER_CHARGER}
- Simulation duration: {config.SIM_MINUTES} minutes
- Time step: {config.DT} seconds
- Battery capacity: {config.BATTERY_KWH} kWh
- Charging power: {config.CHARGING_POWER_KW} kW

## STRATEGIES COMPARED:
1. Baseline (nearest charger)
2. Heuristic (multi-factor scoring)
3. Random selection
4. Oracle (perfect information)
5. Reinforcement Learning (Q-learning)
"""
        
        readme_path = os.path.join(self.dirs['root'], 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def get_path(self, category, filename, strategy=None):
        """Get organized file path"""
        if strategy:
            dir_key = f"{category}_{strategy}"
            if dir_key in self.dirs:
                return os.path.join(self.dirs[dir_key], filename)
        
        # Fallback to main category
        if category in self.dirs:
            return os.path.join(self.dirs[category], filename)
        
        return os.path.join(self.dirs['root'], filename)
    
    def save_configuration(self):
        """Save simulation configuration to JSON"""
        config_dict = {
            'time_settings': {
                'dt_seconds': config.DT,
                'simulation_minutes': config.SIM_MINUTES,
                'simulation_seconds': config.SIM_SECONDS
            },
            'fleet_settings': {
                'num_buses': config.NUM_BUSES,
                'battery_kwh': config.BATTERY_KWH,
                'consumption_kwh_per_km': config.CONSUMPTION_KWH_PER_KM,
                'cruise_speed_kmh': config.CRUISE_SPEED_KMH
            },
            'charging_settings': {
                'initial_soc_range': config.INIT_SOC_RANGE,
                'dock_soc_threshold': config.DOCK_SOC_THRESHOLD,
                'charge_target_soc': config.CHARGE_TARGET_SOC,
                'dock_proximity_km': config.DOCK_PROXIMITY_KM
            },
            'infrastructure': {
                'num_chargers': len(config.CHARGER_LOCATIONS),
                'ports_per_charger': config.PORTS_PER_CHARGER,
                'charging_power_kw': config.CHARGING_POWER_KW,
                'charger_locations': config.CHARGER_LOCATIONS
            },
            'rl_settings': {
                'training_episodes': config.TRAINING_EPISODES,
                'training_episode_seconds': config.TRAINING_EPISODE_SECONDS,
                'state_soc_buckets': config.RL_STATE_SOC_BUCKETS,
                'alpha': config.RL_ALPHA,
                'gamma': config.RL_GAMMA,
                'epsilon_start': config.RL_EPS_START,
                'epsilon_decay': config.RL_EPS_DECAY,
                'epsilon_min': config.RL_MIN_EPS
            },
            'strategies': config.STRATEGIES,
            'output_directory': self.base_dir,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.dirs['logs'], 'simulation_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Configuration saved: {config_path}")
        return config_path

# =============================================
# SIMULATION CLASSES
# =============================================

@dataclass
class ChargingStation:
    """Charging station with comprehensive logging"""
    position_km: Tuple[float, float]
    num_ports: int
    power_kw: float
    ports: List[Optional[int]] = field(init=False)
    waiting_queue: List[int] = field(default_factory=list)
    total_energy_delivered_kwh: float = field(default=0.0, init=False)
    utilization_history: List[float] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        self.ports = [None] * self.num_ports
    
    def has_free_port(self) -> bool:
        return any(p is None for p in self.ports)
    
   
    @property
    def current_load_kw(self) -> float:
        return self.power_kw if self.connected_bus_ids else 0.0

    @property
    def connected_bus_ids(self) -> List[int]:
        return [b for b in self.ports if b is not None]
    
    @property
    def queue_length(self) -> int:
        return len(self.waiting_queue)
    
    def connect(self, bus_id: int) -> bool:
        for i in range(self.num_ports):
            if self.ports[i] is None:
                self.ports[i] = bus_id
                return True
        
        if bus_id not in self.waiting_queue:
            self.waiting_queue.append(bus_id)
        
        return False
    
    def disconnect(self, bus_id: int):
        for i in range(self.num_ports):
            if self.ports[i] == bus_id:
                self.ports[i] = None
                break
        self._fill_from_queue()
    
    def _fill_from_queue(self):
        for i in range(self.num_ports):
            if self.ports[i] is None and self.waiting_queue:
                next_bus = self.waiting_queue.pop(0)
                self.ports[i] = next_bus
    
    def update(self, dt_s: float, buses: List['Bus']):
        for bus_id in self.connected_bus_ids:
            bus = next((b for b in buses if b.bus_id == bus_id), None)
            if bus:
                added_kwh = self.power_kw * dt_s / 3600.0
                current_soc_kwh = bus.soc_fraction * bus.battery_kwh
                max_addable = bus.battery_kwh - current_soc_kwh
                added_kwh = min(added_kwh, max_addable)
                
                if added_kwh > 0:
                    bus.soc_kwh += added_kwh
                    bus.energy_charged_kwh += added_kwh
                    self.total_energy_delivered_kwh += added_kwh
        
        utilization = len(self.connected_bus_ids) / self.num_ports
        self.utilization_history.append(utilization)

@dataclass
class Bus:
    """Electric bus with comprehensive telemetry logging"""
    bus_id: int
    battery_kwh: float
    consumption_kwh_per_km: float
    cruise_speed_kmh: float
    route_waypoints: List[Tuple[float, float]]
    start_offset_km: float = 0.0
    
    # State variables
    soc_kwh: float = field(init=False)
    route_index: int = field(init=False)
    segment_progress_km: float = field(init=False)
    charging_station_id: Optional[int] = field(default=None, init=False)
    is_charging: bool = field(default=False, init=False)
    time_waiting_at_charger_s: float = field(default=0.0, init=False)
    total_energy_consumed_kwh: float = field(default=0.0, init=False)
    energy_charged_kwh: float = field(default=0.0, init=False)
    distance_traveled_km: float = field(default=0.0, init=False)
    
    # Research metrics
    charge_sessions: List[Dict[str, Any]] = field(default_factory=list, init=False)
    soc_history: List[float] = field(default_factory=list, init=False)
    position_history: List[Tuple[float, float]] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        self.soc_kwh = self.battery_kwh * random.uniform(*config.INIT_SOC_RANGE)
        total_route_length = 0
        segment_lengths = []
        
        for i in range(len(self.route_waypoints)):
            a = self.route_waypoints[i]
            b = self.route_waypoints[(i + 1) % len(self.route_waypoints)]
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            segment_lengths.append(seg_len)
            total_route_length += seg_len
        
        offset = self.start_offset_km % total_route_length
        accumulated = 0.0
        for i, seg_len in enumerate(segment_lengths):
            if offset <= accumulated + seg_len:
                self.route_index = i
                self.segment_progress_km = offset - accumulated
                break
            accumulated += seg_len
    
    @property
    def soc_fraction(self) -> float:
        return self.soc_kwh / self.battery_kwh
    
    @property
    def position_km(self) -> Tuple[float, float]:
        a = self.route_waypoints[self.route_index]
        b = self.route_waypoints[(self.route_index + 1) % len(self.route_waypoints)]
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        
        if seg_len == 0:
            return a
        
        t = self.segment_progress_km / seg_len
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t
        )
    
    def update_drive(self, dt_s: float) -> float:
        if self.is_charging:
            return 0.0
        
        # Traffic variation
        speed_multiplier = random.uniform(0.78, 1.22)
        effective_speed = self.cruise_speed_kmh * speed_multiplier
        potential_distance_km = (effective_speed / 3600.0) * dt_s
        
        # Consumption variation
        consumption_multiplier = random.uniform(0.9, 1.1)
        effective_consumption = self.consumption_kwh_per_km * consumption_multiplier
        required_energy_kwh = potential_distance_km * effective_consumption
        
        if self.soc_kwh <= required_energy_kwh:
            actual_distance_km = self.soc_kwh / effective_consumption
            energy_used_kwh = self.soc_kwh
            self.soc_kwh = 0.0
        else:
            actual_distance_km = potential_distance_km
            energy_used_kwh = required_energy_kwh
            self.soc_kwh -= energy_used_kwh
        
        self.distance_traveled_km += actual_distance_km
        self.total_energy_consumed_kwh += energy_used_kwh
        
        remaining_distance = actual_distance_km
        while remaining_distance > 1e-6:
            a = self.route_waypoints[self.route_index]
            b = self.route_waypoints[(self.route_index + 1) % len(self.route_waypoints)]
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            seg_remaining = seg_len - self.segment_progress_km
            
            if remaining_distance <= seg_remaining:
                self.segment_progress_km += remaining_distance
                break
            else:
                self.segment_progress_km = 0.0
                self.route_index = (self.route_index + 1) % len(self.route_waypoints)
                remaining_distance -= seg_remaining
        
        return actual_distance_km
    
    def start_charging(self, station_id: int):
        session = {
            'station_id': station_id,
            'wait_s': self.time_waiting_at_charger_s,
            'soc_at_start': self.soc_fraction,
            'energy_added_kwh': 0.0,
            'duration_s': 0.0,
            'soc_at_end': None
        }
        self.charge_sessions.append(session)
        self.is_charging = True
        self.charging_station_id = station_id
        self.time_waiting_at_charger_s = 0.0
    
    def stop_charging(self):
        if self.charge_sessions:
            self.charge_sessions[-1]['soc_at_end'] = self.soc_fraction
        self.is_charging = False
        self.charging_station_id = None
    
    def update_telemetry(self, timestamp: float):
        self.soc_history.append(self.soc_fraction * 100)
        self.position_history.append(self.position_km)

# =============================================
# REINFORCEMENT LEARNING AGENT
# =============================================

class ResearchRLAgent:
    def __init__(self, num_chargers: int):
        self.num_chargers = num_chargers
        self.eps = config.RL_EPS_START
        self.q_table: Dict[tuple, np.ndarray] = {}
        self.training_history = {
            'episode': [], 'avg_wait': [], 'avg_soc': [], 'epsilon': [],
            'q_table_size': [], 'avg_reward': [], 'decisions_made': [],
            'successful_charges': [], 'failed_charges': []
        }
    
    def state_from_bus(self, bus: 'Bus', sim: 'Simulation') -> tuple:
        soc_bucket = min(config.RL_STATE_SOC_BUCKETS - 1, 
                        int(bus.soc_fraction * config.RL_STATE_SOC_BUCKETS))
        
        _, nearest_dist = sim._nearest_charger(bus.position_km)
        dist_bucket = min(config.RL_STATE_DIST_BUCKETS - 1, int(nearest_dist))
        
        charger_states = []
        for ch in sim.chargers:
            free_port = 1 if ch.has_free_port() else 0
            queue_level = min(3, len(ch.waiting_queue))
            utilization = len(ch.connected_bus_ids) / ch.num_ports
            charger_states.extend([free_port, queue_level, int(utilization * 3)])
        
        time_factor = int((sim.t_s % 86400) / 21600)
        
        return (soc_bucket, dist_bucket, time_factor) + tuple(charger_states)
    
    def ensure_state(self, state: tuple):
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(0.0, 0.5, self.num_chargers)
    
    def select_action(self, state: tuple, explore: bool = True) -> int:
        self.ensure_state(state)
        
        if explore and random.random() < self.eps:
            return random.randint(0, self.num_chargers - 1)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        self.ensure_state(state)
        self.ensure_state(next_state)
        
        old_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        target = reward + config.RL_GAMMA * max_next_q
        self.q_table[state][action] += config.RL_ALPHA * (target - old_q)
    
    def decay_epsilon(self):
        self.eps = max(config.RL_MIN_EPS, self.eps * config.RL_EPS_DECAY)
    
    def record_training(self, episode, avg_wait, avg_soc, avg_reward, 
                       decisions, successful, failed):
        self.training_history['episode'].append(episode)
        self.training_history['avg_wait'].append(avg_wait)
        self.training_history['avg_soc'].append(avg_soc)
        self.training_history['epsilon'].append(self.eps)
        self.training_history['q_table_size'].append(len(self.q_table))
        self.training_history['avg_reward'].append(avg_reward)
        self.training_history['decisions_made'].append(decisions)
        self.training_history['successful_charges'].append(successful)
        self.training_history['failed_charges'].append(failed)
    
    def save_training_data(self, output_manager: OutputManager):
        """Save training data using OutputManager"""
        # Save training history
        df = pd.DataFrame(self.training_history)
        csv_path = output_manager.get_path('logs', 'rl_training_history.csv')
        df.to_csv(csv_path, index=False)
        
        # Save Q-table stats
        q_table_stats = {
            'num_states': len(self.q_table),
            'state_examples': list(self.q_table.keys())[:5],
            'epsilon': self.eps
        }
        stats_path = output_manager.get_path('logs', 'rl_qtable_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(q_table_stats, f, indent=2)
        
        print(f"  RL training data saved to {csv_path}")
        return df

# =============================================
# SIMULATION CLASS
# =============================================

class Simulation:
    def __init__(self, strategy: str = "baseline", seed: Optional[int] = None, 
                 rl_agent: Optional[ResearchRLAgent] = None):
        self.strategy = strategy
        self.rl_agent = rl_agent
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create charging infrastructure
        self.chargers = [
            ChargingStation(pos, config.PORTS_PER_CHARGER, config.CHARGING_POWER_KW)
            for pos in config.CHARGER_LOCATIONS
        ]
        
        # Create bus fleet with staggered starts
        self.buses = []
        total_route_length = sum(
            math.hypot(
                config.ROUTE_WAYPOINTS[i][0] - config.ROUTE_WAYPOINTS[(i + 1) % len(config.ROUTE_WAYPOINTS)][0],
                config.ROUTE_WAYPOINTS[i][1] - config.ROUTE_WAYPOINTS[(i + 1) % len(config.ROUTE_WAYPOINTS)][1]
            ) for i in range(len(config.ROUTE_WAYPOINTS))
        )
        
        offsets = np.linspace(0, total_route_length, config.NUM_BUSES, endpoint=False)
        
        for i in range(config.NUM_BUSES):
            bus = Bus(
                bus_id=i,
                battery_kwh=config.BATTERY_KWH,
                consumption_kwh_per_km=config.CONSUMPTION_KWH_PER_KM,
                cruise_speed_kmh=config.CRUISE_SPEED_KMH,
                route_waypoints=config.ROUTE_WAYPOINTS,
                start_offset_km=float(offsets[i])
            )
            self.buses.append(bus)
        
        # Initialize history
        self.t_s = 0
        self.timestamps = []
        self.soc_history = [[] for _ in range(config.NUM_BUSES)]
        self.position_history = []
        self.load_history = []
        self.queue_history = []
        self.charger_utilization = [[] for _ in range(len(self.chargers))]
        self.bus_positions_history = []
        
        self.metrics = {}
    
    def _nearest_charger(self, position: Tuple[float, float]) -> Tuple[int, float]:
        best_idx = 0
        best_dist = float('inf')
        for i, charger in enumerate(self.chargers):
            dist = math.hypot(
                position[0] - charger.position_km[0],
                position[1] - charger.position_km[1]
            )
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx, best_dist
    
    def _choose_charger(self, bus: Bus) -> int:
        if self.strategy == "baseline":
            idx, _ = self._nearest_charger(bus.position_km)
            return idx
        
        elif self.strategy == "heuristic":
            best_score = float('inf')
            best_idx = 0
            for i, charger in enumerate(self.chargers):
                distance = math.hypot(
                    bus.position_km[0] - charger.position_km[0],
                    bus.position_km[1] - charger.position_km[1]
                )
                queue_penalty = charger.queue_length * 80
                dist_penalty = distance * 30
                busy_penalty = 150 if not charger.has_free_port() else 0
                score = queue_penalty + dist_penalty + busy_penalty
                if score < best_score:
                    best_score = score
                    best_idx = i
            return best_idx
        
        elif self.strategy == "random":
            return random.randint(0, len(self.chargers) - 1)

        elif self.strategy == "oracle":
            best_idx = 0
            best_score = float('inf')
            for i, charger in enumerate(self.chargers):
                queue = charger.queue_length
                dist = math.hypot(
                    bus.position_km[0] - charger.position_km[0],
                    bus.position_km[1] - charger.position_km[1]
                )
                score = queue * 50 + dist * 2
                if score < best_score:
                    best_score = score
                    best_idx = i
            return best_idx
        
        elif self.strategy == "rl":
            if self.rl_agent is None:
                return random.randint(0, len(self.chargers) - 1)
            state = self.rl_agent.state_from_bus(bus, self)
            action = self.rl_agent.select_action(state, explore=False)
            if action is None or not (0 <= action < len(self.chargers)):
                return random.randint(0, len(self.chargers) - 1)
            return action
        
        idx, _ = self._nearest_charger(bus.position_km)
        return idx
    
    def step(self, dt_s: float):
        # Update chargers
        for charger in self.chargers:
            charger.update(dt_s, self.buses)
        
        # Update buses
        for bus in self.buses:
            bus.update_drive(dt_s)
        
        # Update waiting times
        for charger in self.chargers:
            for bus_id in charger.waiting_queue:
                bus = next((b for b in self.buses if b.bus_id == bus_id), None)
                if bus:
                    bus.time_waiting_at_charger_s += dt_s
        
        # Charging decisions
        for bus in self.buses:
            if bus.is_charging:
                if bus.soc_fraction >= config.CHARGE_TARGET_SOC:  # Use config value  # Charging target
                    charger = self.chargers[bus.charging_station_id]
                    charger.disconnect(bus.bus_id)
                    bus.stop_charging()
            else:
                if bus.soc_fraction < config.DOCK_SOC_THRESHOLD:
                    chosen_idx = self._choose_charger(bus)
                    chosen_charger = self.chargers[chosen_idx]
                    distance = math.hypot(
                        bus.position_km[0] - chosen_charger.position_km[0],
                        bus.position_km[1] - chosen_charger.position_km[1]
                    )
                    if distance <= config.DOCK_PROXIMITY_KM:
                        connected = chosen_charger.connect(bus.bus_id)
                        if connected:
                            bus.start_charging(chosen_idx)
        
        # Fill empty ports from queue
        for charger in self.chargers:
            charger._fill_from_queue()
        
        # Record state
        self.t_s += dt_s
        self.timestamps.append(self.t_s)
        
        for i, bus in enumerate(self.buses):
            self.soc_history[i].append(bus.soc_fraction * 100)
            bus.update_telemetry(self.t_s)
        
        current_positions = [bus.position_km for bus in self.buses]
        self.position_history.append(current_positions)
        self.bus_positions_history.append(current_positions)
        
        total_load = sum(charger.current_load_kw for charger in self.chargers)
        self.load_history.append(total_load)
        
        total_queue = sum(charger.queue_length for charger in self.chargers)
        self.queue_history.append(total_queue)
        
        for i, charger in enumerate(self.chargers):
            utilization = len(charger.connected_bus_ids) / charger.num_ports
            self.charger_utilization[i].append(utilization)
    
    def run(self, total_seconds: int, dt_s: int):
        steps = total_seconds // dt_s
        print(f"  Running {self.strategy} strategy: {steps} steps")
        
        for step in range(steps):
            if (step + 1) % max(1, steps // 10) == 0:

                print(f"    Progress: {100 * (step + 1) // steps}%")
            self.step(dt_s)
        
        self._calculate_metrics()
        return self.metrics
    
    def _calculate_metrics(self):
        avg_wait = np.mean([bus.time_waiting_at_charger_s for bus in self.buses])
        max_wait = max(bus.time_waiting_at_charger_s for bus in self.buses)
        total_wait = sum(bus.time_waiting_at_charger_s for bus in self.buses)
        
        soc_means = [np.mean(soc) for soc in self.soc_history]
        avg_soc = np.mean(soc_means)
        min_soc = min(min(soc) for soc in self.soc_history)
        
        total_energy_used = sum(bus.total_energy_consumed_kwh for bus in self.buses)
        total_energy_charged = sum(bus.energy_charged_kwh for bus in self.buses)
        total_distance = sum(bus.distance_traveled_km for bus in self.buses)
        
        buses_charged = sum(1 for bus in self.buses if bus.energy_charged_kwh > 0)
        total_sessions = sum(len(bus.charge_sessions) for bus in self.buses)
        
        avg_queue = np.mean(self.queue_history) if self.queue_history else 0
        max_queue = max(self.queue_history) if self.queue_history else 0
        avg_load = np.mean(self.load_history) if self.load_history else 0
        peak_load = max(self.load_history) if self.load_history else 0
        # In Simulation._calculate_metrics(), add:
        dangerously_low_buses = sum(1 for bus in self.buses if bus.soc_fraction < 0.20)
        self.metrics['buses_below_20_percent'] = dangerously_low_buses
        charger_metrics = []
        for i, charger in enumerate(self.chargers):
            charger_metrics.append({
                'id': i,
                'energy_delivered_kwh': charger.total_energy_delivered_kwh,
                'avg_utilization': np.mean(charger.utilization_history) if charger.utilization_history else 0,
                'max_utilization': max(charger.utilization_history) if charger.utilization_history else 0,
                'total_queue_time': sum(bus.time_waiting_at_charger_s 
                                      for bus in self.buses 
                                      if bus.bus_id in charger.waiting_queue)
            })
        
        self.metrics = {
            'strategy': self.strategy,
            'simulation_time_s': self.t_s,
            'avg_wait_time_s': avg_wait,
            'max_wait_time_s': max_wait,
            'total_wait_time_s': total_wait,
            'avg_soc_percent': avg_soc,
            'min_soc_percent': min_soc,
            'soc_std_percent': np.std(soc_means),
            'total_energy_used_kwh': total_energy_used,
            'total_energy_charged_kwh': total_energy_charged,
            'energy_efficiency_kwh_per_km': total_energy_used / max(0.001, total_distance),
            'buses_that_charged': buses_charged,
            'total_charge_sessions': total_sessions,
            'total_distance_km': total_distance,
            'avg_distance_per_bus_km': total_distance / len(self.buses),
            'avg_queue_length': avg_queue,
            'max_queue_length': max_queue,
            'avg_load_kw': avg_load,
            'peak_load_kw': peak_load,
            'charger_metrics': charger_metrics,
            'timestamps': self.timestamps,
            'soc_history': self.soc_history,
            'load_history': self.load_history,
            'queue_history': self.queue_history,
            'position_history': self.position_history,
            'bus_positions_history': self.bus_positions_history
        }
    
    def export_detailed_data(self, output_manager: OutputManager):
        """Export data using organized OutputManager"""
        # Time series data
        timeseries_file = output_manager.get_path('data', f'timeseries_{self.strategy}.csv', self.strategy)
        with open(timeseries_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['timestamp_s', 'total_load_kw', 'total_queue']
            headers.extend([f'bus_{i}_soc_percent' for i in range(config.NUM_BUSES)])
            headers.extend([f'charger_{i}_utilization' for i in range(len(self.chargers))])
            writer.writerow(headers)
            
            for step_idx in range(len(self.timestamps)):
                t = self.timestamps[step_idx]
                load = self.load_history[step_idx]
                queue = self.queue_history[step_idx]
                socs = [self.soc_history[i][step_idx] for i in range(config.NUM_BUSES)]
                utils = [self.charger_utilization[i][step_idx] for i in range(len(self.chargers))]
                row = [t, load, queue] + socs + utils
                writer.writerow(row)
        
        # Bus-specific data
        buses_file = output_manager.get_path('data', f'buses_{self.strategy}.csv', self.strategy)
        with open(buses_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bus_id', 'total_wait_s', 'energy_used_kwh',
                           'energy_charged_kwh', 'distance_km', 'avg_soc_percent',
                           'min_soc_percent', 'charge_sessions', 'avg_soc_history'])
            
            for bus in self.buses:
                bus_soc_history = self.soc_history[bus.bus_id]
                writer.writerow([
                    bus.bus_id,
                    f"{bus.time_waiting_at_charger_s:.2f}",
                    f"{bus.total_energy_consumed_kwh:.2f}",
                    f"{bus.energy_charged_kwh:.2f}",
                    f"{bus.distance_traveled_km:.2f}",
                    f"{np.mean(bus_soc_history):.2f}",
                    f"{min(bus_soc_history):.2f}",
                    len(bus.charge_sessions),
                    ';'.join([f"{soc:.2f}" for soc in bus_soc_history])
                ])
        
        print(f"  Data exported: {timeseries_file}")

# =============================================
# VISUALIZATION FUNCTIONS
# =============================================

class VisualizationManager:
    """Manages all visualization creation"""
    
    @staticmethod
    def create_simulation_visualization(sim: Simulation, output_manager: OutputManager):
        """Create static visualization for a strategy"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # SOC over time
            for i in range(min(4, config.NUM_BUSES)):
                axes[0, 0].plot(sim.timestamps, sim.soc_history[i], label=f'Bus {i}', alpha=0.8)
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('SOC (%)')
            axes[0, 0].set_title('State of Charge Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Load
            axes[0, 1].plot(sim.timestamps, sim.load_history, 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Total Load (kW)')
            axes[0, 1].set_title('System Charging Load')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Queue length
            axes[1, 0].plot(sim.timestamps, sim.queue_history, 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Total Queue Length')
            axes[1, 0].set_title('Charging Queue Dynamics')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Utilization
            for i in range(len(sim.chargers)):
                util = [u for u in sim.charger_utilization[i][:len(sim.timestamps)]]
                axes[1, 1].plot(sim.timestamps[:len(util)], util, label=f'Charger {i}')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Utilization')
            axes[1, 1].set_title('Charger Utilization')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'{sim.strategy.upper()} Strategy - System Dynamics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = output_manager.get_path('plots', f'system_dynamics_{sim.strategy}.pdf', sim.strategy)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  System dynamics plot: {plot_path}")
            
        except Exception as e:
            print(f"  Visualization failed: {e}")
    
    @staticmethod
    def create_animation(sim: Simulation, output_manager: OutputManager):
        """Create animated GIF of simulation"""
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(-6, 23)
            ax.set_ylim(-1, 55)
            ax.set_xlabel('X Position (km)', fontsize=12)
            ax.set_ylabel('Y Position (km)', fontsize=12)
            
            # Plot route
            route_x = [p[0] for p in config.ROUTE_WAYPOINTS]
            route_y = [p[1] for p in config.ROUTE_WAYPOINTS]
            ax.plot(route_x, route_y, 'k--', alpha=0.5, linewidth=2, label='Bus Route')
            
            # Plot chargers
            for i, pos in enumerate(config.CHARGER_LOCATIONS):
                ax.plot(pos[0], pos[1], 's', markersize=15, color='red', label='Charger' if i == 0 else "")
                ax.text(pos[0], pos[1] + 1.5, f'Charger {i}', fontsize=12, fontweight='bold', ha='center', color='red')
            
            # Create scatter points for buses
            colors = plt.cm.tab10(np.linspace(0, 1, config.NUM_BUSES))
            scatters = []
            for i in range(config.NUM_BUSES):
                s = ax.scatter([], [], s=120, color=colors[i], edgecolors='black', linewidth=1.5, label=f'Bus {i}')
                scatters.append(s)
            
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
            
            def update(frame):
                positions = sim.bus_positions_history[frame]
                socs = [s[frame] for s in sim.soc_history]
                
                for i, (scatter, pos, soc) in enumerate(zip(scatters, positions, socs)):
                    scatter.set_offsets([pos])
                    # Color by SOC
                    if soc < 40:
                        scatter.set_color('red')
                    elif soc < 60:
                        scatter.set_color('orange')
                    else:
                        scatter.set_color('green')
                
                total_load = sim.load_history[frame]
                avg_soc = np.mean(socs)
                ax.set_title(f'Time: {frame*config.DT//60} min | '
                            f'Avg SOC: {avg_soc:.1f}% | '
                            f'Load: {total_load:.0f} kW | '
                            f'Strategy: {sim.strategy.upper()}', fontsize=14, fontweight='bold')
                return scatters
            
            anim = animation.FuncAnimation(fig, update, frames=len(sim.bus_positions_history), 
                                          interval=150, blit=False, repeat=True)
            
            # Save animation
            gif_path = output_manager.get_path('animations', f'animation_{sim.strategy}.gif', sim.strategy)
            anim.save(gif_path, writer=PillowWriter(fps=8))
            plt.close()
            print(f"  Animation saved: {gif_path}")
            
        except Exception as e:
            print(f"  Animation failed: {e}")
    @staticmethod
    def create_training_plots(training_history, output_manager):
        """Create RL training progress plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Wait time progression
            axes[0, 0].plot(training_history['episode'], training_history['avg_wait'], 
                          'b-', linewidth=2, alpha=0.7)
            axes[0, 0].set_xlabel('Training Episode')
            axes[0, 0].set_ylabel('Average Wait Time (s)')
            axes[0, 0].set_title('RL Training: Wait Time Progression')
            axes[0, 0].grid(True, alpha=0.3)
            
            # SOC progression
            axes[0, 1].plot(training_history['episode'], training_history['avg_soc'], 
                          'g-', linewidth=2, alpha=0.7)
            axes[0, 1].axhline(y=60, color='r', linestyle='--', alpha=0.5, label='Target SOC')
            axes[0, 1].set_xlabel('Training Episode')
            axes[0, 1].set_ylabel('Average SOC (%)')
            axes[0, 1].set_title('RL Training: SOC Progression')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Epsilon decay
            axes[1, 0].plot(training_history['episode'], training_history['epsilon'], 
                          'r-', linewidth=2)
            axes[1, 0].set_xlabel('Training Episode')
            axes[1, 0].set_ylabel('Exploration Rate (ε)')
            axes[1, 0].set_title('RL Training: Exploration Rate Decay')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
            
            # Q-table size growth
            axes[1, 1].plot(training_history['episode'], training_history['q_table_size'], 
                          'm-', linewidth=2)
            axes[1, 1].set_xlabel('Training Episode')
            axes[1, 1].set_ylabel('Number of States')
            axes[1, 1].set_title('RL Training: State Space Exploration')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle('Reinforcement Learning Training Progress', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = output_manager.get_path('plots', 'rl_training_progress.pdf')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Training plots saved: {plot_path}")
            
        except Exception as e:
            print(f"  Training plots failed: {e}")
    
# =============================================
# TRAINING FUNCTIONS
# =============================================

def train_rl_agent(output_manager: OutputManager):
    """Train RL agent with comprehensive logging"""
    print("\n" + "="*70)
    print("COMPREHENSIVE RL TRAINING WITH DETAILED LOGGING")
    print(f"Episodes: {config.TRAINING_EPISODES}, Duration: {config.TRAINING_EPISODE_SECONDS/60:.0f} min each")
    print("="*70)
    
    agent = ResearchRLAgent(len(config.CHARGER_LOCATIONS))
    
    for episode in range(config.TRAINING_EPISODES):
        # Create fresh simulation for training
        sim = Simulation(strategy='rl', seed=config.SEED + episode * 100, rl_agent=agent)
        
        # Lower SOC for training to force charging decisions
        for bus in sim.buses:
           
            bus.soc_kwh = bus.battery_kwh * random.uniform(*config.INIT_SOC_RANGE)

        episode_rewards = []
        episode_decisions = 0
        successful_charges = 0
        failed_charges = 0
        
        steps = config.TRAINING_EPISODE_SECONDS // config.DT
        for step in range(steps):
            # Make RL decisions
            step_decisions = []
            for bus in sim.buses:
                if bus.soc_fraction < config.DOCK_SOC_THRESHOLD and not bus.is_charging:
                    state = agent.state_from_bus(bus, sim)
                    action = agent.select_action(state, explore=True)
                    
                    step_decisions.append({
                        'bus': bus,
                        'state': state,
                        'action': action,
                        'prev_soc': bus.soc_fraction,
                        'prev_wait': bus.time_waiting_at_charger_s
                    })
                    episode_decisions += 1
            
            # Execute step
            sim.step(config.DT)
            
            # Process rewards
            for d in step_decisions:
                bus = d['bus']
                reward = 0.0
                
                # Check outcome
                if bus.is_charging and bus.charging_station_id == d['action']:
                    successful_charges += 1
                    reward += 100.0
                    reward += (bus.soc_fraction - d['prev_soc']) * 300.0
                if bus.soc_fraction < 0.25:  # Penalize low SOC (25% safety margin)
                    safety_penalty = -200 * (0.25 - bus.soc_fraction)  # Strong penalty
                    reward += safety_penalty
                elif any(bus.bus_id in ch.waiting_queue for ch in sim.chargers):
                    failed_charges += 1
                    reward -= 50.0
                
                # Distance penalty
                charger_pos = sim.chargers[d['action']].position_km
                dist = math.hypot(bus.position_km[0] - charger_pos[0],
                                 bus.position_km[1] - charger_pos[1])
                reward -= dist * 20.0
                
                # SOC maintenance
                if bus.soc_fraction < 0.3:
                    reward -= 100.0 * (0.3 - bus.soc_fraction)
                
                # Update agent
                next_state = agent.state_from_bus(bus, sim)
                agent.update(d['state'], d['action'], reward, next_state)
                episode_rewards.append(reward)
        
        # Calculate episode metrics
        avg_wait = np.mean([b.time_waiting_at_charger_s for b in sim.buses])
        avg_soc = np.mean([b.soc_fraction * 100 for b in sim.buses])
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        agent.record_training(episode + 1, avg_wait, avg_soc, avg_reward,
                             episode_decisions, successful_charges, failed_charges)
        
        # Progress reporting
        if (episode + 1) % 20 == 0 or episode < 5:
            print(f"  Episode {episode+1}/{config.TRAINING_EPISODES}: "
                  f"Wait={avg_wait:.1f}s, SOC={avg_soc:.1f}%, "
                  f"eps={agent.eps:.3f}, States={len(agent.q_table)}, "
                  f"Reward={avg_reward:.1f}")
        
        agent.decay_epsilon()
    
    # Save training data
    agent.save_training_data(output_manager)
    
    # Create training plots
    VisualizationManager.create_training_plots(agent.training_history, output_manager)
    
    print(f"\nTraining complete. Final epsilon: {agent.eps:.3f}, States learned: {len(agent.q_table)}")
    return agent

# =============================================
# MAIN STUDY EXECUTION
# =============================================

def run_complete_study(num_simulations=30):
    """Run complete study with organized outputs"""
    # Create output manager
    output_manager = OutputManager()
    output_manager.save_configuration()
    
    print("\n" + "="*80)
    print("COMPLETE IEEE STUDY - SHENZHEN ELECTRIC BUS CHARGING OPTIMIZATION")
    print("="*80)
    print(f"Output directory: {output_manager.base_dir}")
    print(f"Simulations per strategy: {num_simulations}")
    print("="*80)
    
    # 1. Train RL agent
    print("\n[1/5] Training RL agent with comprehensive logging...")
    rl_agent = train_rl_agent(output_manager)
    
    # 2. Run multiple simulations
    print(f"\n[2/5] Running {num_simulations} simulations per strategy...")
    all_results = {strat: {'waits': [], 'socs': [], 'details': []} for strat in config.STRATEGIES}
    
    for run_idx in range(num_simulations):
        seed = config.SEED + run_idx * 100
        if (run_idx + 1) % 5 == 0:
            print(f"  Run {run_idx+1}/{num_simulations} (Seed: {seed})")
        
        for strategy in config.STRATEGIES:
            agent_to_use = rl_agent if strategy == 'rl' else None
            sim = Simulation(strategy=strategy, seed=seed, rl_agent=agent_to_use)
            
            # Run simulation
            metrics = sim.run(config.SIM_SECONDS, config.DT)
            
            # Export data
            sim.export_detailed_data(output_manager)
            
            # Create visualizations
            VisualizationManager.create_simulation_visualization(sim, output_manager)
            if run_idx == 0:  # only first run
                VisualizationManager.create_animation(sim, output_manager)

            
            
            # Store results
            all_results[strategy]['waits'].append(metrics['avg_wait_time_s'])
            all_results[strategy]['socs'].append(metrics['avg_soc_percent'])
            all_results[strategy]['details'].append(metrics)
    
    # 3. Statistical analysis
    print("\n[3/5] Performing comprehensive statistical analysis...")
    stats_summary = calculate_complete_statistics(all_results, output_manager)
    
    # 4. Create comparison plots
    print("\n[4/5] Creating publication-ready visualizations...")
    create_comparison_plots(stats_summary, all_results, output_manager)
    
    # 5. Generate paper materials
    print("\n[5/5] Generating IEEE paper materials...")
    generate_paper_materials(stats_summary, output_manager)
    
    print(f"\n{'='*80}")
    print("STUDY COMPLETE - ALL OUTPUTS GENERATED")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {os.path.abspath(output_manager.base_dir)}")
    
    return stats_summary

# =============================================
# STATISTICAL ANALYSIS
# =============================================

def calculate_complete_statistics(all_results, output_manager):
    """Calculate comprehensive statistics with confidence intervals"""
    stats_summary = {}
    
    print("\nSTATISTICAL RESULTS (Mean ± 95% CI)")
    print("-"*70)
    print(f"{'Strategy':12} {'Wait Time (s)':20} {'SOC (%)':15}")
    print("-"*70)
    
    for strategy in config.STRATEGIES:
        waits = np.array(all_results[strategy]['waits'])
        socs = np.array(all_results[strategy]['socs'])
        
        mean_wait = np.mean(waits)
        std_wait = np.std(waits)
        mean_soc = np.mean(socs)
        std_soc = np.std(socs)
        
        # 95% confidence intervals
        n = len(waits)
        ci_wait = 1.96 * std_wait / np.sqrt(n)
        ci_soc = 1.96 * std_soc / np.sqrt(n)
        
        stats_summary[strategy] = {
            'mean_wait': mean_wait,
            'std_wait': std_wait,
            'ci_wait': ci_wait,
            'mean_soc': mean_soc,
            'std_soc': std_soc,
            'ci_soc': ci_soc,
            'waits': waits.tolist(),
            'socs': socs.tolist(),
            'n': n
        }
        
        print(f"{strategy.upper():12} {mean_wait:6.1f} ± {ci_wait:4.1f}        "
              f"{mean_soc:5.1f} ± {ci_soc:3.1f}")
    
    # Save statistical results
    stats_df = pd.DataFrame([
        {
            'strategy': s,
            'mean_wait': stats_summary[s]['mean_wait'],
            'ci_wait': stats_summary[s]['ci_wait'],
            'mean_soc': stats_summary[s]['mean_soc'],
            'ci_soc': stats_summary[s]['ci_soc'],
            'n': stats_summary[s]['n']
        }
        for s in config.STRATEGIES
    ])
    
    stats_path = output_manager.get_path('tables', 'statistical_results.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistical results saved: {stats_path}")
    
    return stats_summary
def create_comparison_plots(stats_summary, all_results, output_manager):
    """Create strategy comparison plots"""
    try:
        # Main comparison figure
        fig, ax = plt.subplots(figsize=(12, 8))
        strategies = list(stats_summary.keys())
        means = [stats_summary[s]['mean_wait'] for s in strategies]
        cis = [stats_summary[s]['ci_wait'] for s in strategies]
        
        colors = ['gray', 'skyblue', 'lightgreen', 'orange', 'purple']
        x_pos = np.arange(len(strategies))
        
        bars = ax.bar(x_pos, means, yerr=cis, capsize=10, alpha=0.8,
                     color=colors, edgecolor='black', linewidth=1.5)
        
        for i, (mean, ci) in enumerate(zip(means, cis)):
            ax.text(i, mean + ci + 50, f'{mean:.0f}±{ci:.0f}s', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Charging Strategy', fontsize=12)
        ax.set_ylabel('Average Wait Time (seconds)', fontsize=12)
        ax.set_title('Comparison of EV Charging Strategies\n(Mean ± 95% Confidence Interval)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.upper() for s in strategies], fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = output_manager.get_path('plots', 'strategy_comparison.pdf')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Strategy comparison plot: {plot_path}")
        
        # SOC vs Wait time scatter
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for strategy in strategies:
            waits = all_results[strategy]['waits']
            socs = all_results[strategy]['socs']
            ax2.scatter(waits, socs, alpha=0.6, s=50, label=strategy.upper())
        
        ax2.set_xlabel('Average Wait Time (s)', fontsize=12)
        ax2.set_ylabel('Average SOC (%)', fontsize=12)
        ax2.set_title('Wait Time vs SOC Trade-off Across Strategies', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        scatter_path = output_manager.get_path('plots', 'wait_vs_soc_scatter.pdf')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Wait vs SOC scatter plot: {scatter_path}")
        
        # Box plot comparison
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        wait_data = [all_results[s]['waits'] for s in strategies]
        box = ax3.boxplot(wait_data, patch_artist=True, 
                         labels=[s.upper() for s in strategies])
        
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Wait Time (s)', fontsize=12)
        ax3.set_title('Distribution of Wait Times Across Strategies', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        
        boxplot_path = output_manager.get_path('plots', 'wait_time_boxplot.pdf')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Wait time box plot: {boxplot_path}")
        
    except Exception as e:
        print(f"  Comparison plots failed: {e}")
# =============================================
# VISUALIZATION MANAGER CONTINUED
# =============================================

def generate_paper_materials(stats_summary, output_manager):
    """Generate IEEE paper materials"""
    try:
        # Create LaTeX table
        latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Performance comparison of charging strategies (mean $\\pm$ 95\\% CI, $N=30$)}
\\label{tab:results}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Strategy} & \\textbf{Wait Time (s)} & \\textbf{SOC (\\%)} \\\\
\\midrule
"""
        
        for strategy in config.STRATEGIES:
            s = stats_summary[strategy]
            latex_content += f"{strategy.capitalize():10} & ${s['mean_wait']:.1f} \\pm {s['ci_wait']:.1f}$ & ${s['mean_soc']:.1f} \\pm {s['ci_soc']:.1f}$ \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        latex_path = output_manager.get_path('paper', 'ieee_results_table.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        
        print(f"  LaTeX table saved: {latex_path}")
        
        # Create summary report
        summary = f"""IEEE STUDY SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M')}

SIMULATION PARAMETERS:
- Buses: {config.NUM_BUSES}
- Charging stations: {len(config.CHARGER_LOCATIONS)}
- Ports per station: {config.PORTS_PER_CHARGER}
- Simulation duration: {config.SIM_MINUTES} minutes
- Time step: {config.DT} seconds

RESULTS SUMMARY:
"""
        for strategy in config.STRATEGIES:
            s = stats_summary[strategy]
            improvement = ""
            if strategy != 'baseline':
                baseline_wait = stats_summary['baseline']['mean_wait']
                improvement = f" ({(baseline_wait - s['mean_wait']) / baseline_wait * 100:.1f}% improvement)"
            
            summary += f"- {strategy.upper():12}: {s['mean_wait']:.1f} ± {s['ci_wait']:.1f} s{improvement}\n"
        
        summary_path = output_manager.get_path('paper', 'study_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"  Study summary saved: {summary_path}")
        
    except Exception as e:
        print(f"  Paper materials failed: {e}")

# =============================================
# MAIN EXECUTION
# =============================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("SHENZHEN ELECTRIC BUS CHARGING OPTIMIZATION - CLEAN VERSION")
    print("="*80)
    print("IEEE-ready simulation with organized output structure")
    print("="*80)
    
    # Run complete study
    try:
        stats = run_complete_study(num_simulations=30)
        
        print("\n" + "="*80)
        print("STUDY COMPLETE!")
        print("="*80)
        print("\nYour organized outputs include:")
        print("  /data/          - CSV files for each strategy")
        print("  /plots/         - PDF visualizations")  
        print("  /animations/    - GIF animations")
        print("  /paper_materials/ - LaTeX tables and summaries")
        print("  /tables/        - Statistical results")
        print("  /logs/          - Configuration and training logs")
        print("\nCheck the README.md for complete directory structure.")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n\nError during simulation: {e}")
        import traceback
        traceback.print_exc()