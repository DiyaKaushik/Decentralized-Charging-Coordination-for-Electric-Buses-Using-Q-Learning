# Electric Bus Charging Coordination with Q-Learning

![Electric Bus Simulation](results/figures/Comparison_of_EV_Charging_Strategies.png)

Official implementation of **"Decentralized Charging Coordination for Electric Buses: A Q-Learning Approach"** - a reinforcement learning solution for optimizing electric bus charging in urban transit systems.

## 📋 Overview

This project addresses the critical challenge of coordinating charging for electric bus fleets operating on fixed urban routes. By formulating the problem as a Markov Decision Process and applying tabular Q-learning, we develop a decentralized policy that significantly reduces charging wait times while maintaining safe battery levels.

## 🚀 Key Results

| Metric | Baseline | Tuned Heuristic | **RL Agent** | Improvement |
|--------|----------|-----------------|--------------|-------------|
| **Avg. Wait Time** | 729.2 ± 187.0 s | 373.3 ± 102.2 s | **178.8 ± 82.0 s** | **75.5%** vs Baseline |
| **Avg. State of Charge** | 55.1% | 54.9% | **42.2%** | More efficient utilization |
| **Statistical Significance** | p < 0.001 | p < 0.001 | - | Robust performance |

## 🏗️ System Architecture

The simulation models Shenzhen Bus Line 303 with:
- **12 BYD K9 electric buses** (350 kWh battery capacity)
- **5 charging stations** (dual 120 kW ports each)
- **84.2 km round-trip route**
- **180-minute simulation** with 5-second time steps

## 🧠 Reinforcement Learning Approach

### State Space
```
s = (SOC_bucket, dist_bucket, time_factor, c₁, c₂, c₃, c₄, c₅)
```
- **SOC_bucket**: Battery state-of-charge (6 discrete levels)
- **dist_bucket**: Distance to nearest charger (5 levels)
- **time_factor**: Time-of-day indicator (4 periods)
- **cᵢ**: Charger status triple (availability, queue, utilization)

### Action Space
```
A = {0, 1, 2, 3, 4}  # Select one of 5 charging stations
```

### Reward Function
```python
R(s,a,s') = 100·I{connection} + 300·ΔSOC - 20·distance
           - penalty(SOC) - penalty(queue)
```

## 📁 Project Structure

```
electric-bus-rl/
├── src/                    # Simulation source code
│   ├── bus.py             # Bus dynamics and energy model
│   ├── charging_station.py # Charging infrastructure
│   ├── simulation.py      # Main simulation loop
│   ├── rl_agent.py       # Q-learning implementation
│   └── config.py         # Simulation parameters
├── results/               # Key outputs
│   ├── figures/          # Generated plots
│   └── tables/           # Statistical results
├── paper/                 # Paper materials
│   └── latex/            # LaTeX source files
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## 🛠️ Installation & Usage

### Prerequisites
```bash
python >= 3.8
pip install -r requirements.txt
```

### Quick Start
```python
# Run a simulation with RL agent
from src.simulation import Simulation
from src.rl_agent import RlAgent

sim = Simulation(config_path='config/shenzhen_line_303.json')
agent = RlAgent()
results = sim.run(agent, duration_minutes=180)
```

### Training the Agent
```bash
python scripts/train_agent.py --episodes 200 --save_path models/q_table.npy
```

### Evaluating Strategies
```bash
python scripts/run_comparison.py --strategies baseline heuristic random rl
```

## 📊 Performance Comparison

![Training Progress](results/figures/RL_Training_Wait_Time_Progression.png)
![SOC Dynamics](results/figures/State_of_Charge_Over_Time.png)

**Key Insights:**
1. **Anticipatory Charging**: RL agent initiates charging at 35-38% SOC (vs 40% threshold)
2. **Load Balancing**: Distributes buses across stations to reduce congestion
3. **Safety-Conscious**: Maintains all buses above 30% SOC safety limit

## 📈 Results Analysis

The Q-learning agent demonstrates:
- **75.5% reduction** in average wait time compared to nearest-charger baseline
- **52.1% improvement** over carefully tuned heuristic
- **Lower average SOC** (42.2% vs ~55%) indicating efficient battery utilization
- **Reduced variability** in wait times (smaller confidence intervals)

## 🔧 Configuration

Key simulation parameters in `src/config.py`:
```python
SIMULATION_CONFIG = {
    'num_buses': 12,
    'num_stations': 5,
    'ports_per_station': 2,
    'battery_capacity': 350,  # kWh
    'charging_power': 120,    # kW per port
    'consumption_rate': 1.65, # kWh/km
    'route_length': 84.2,     # km
    'soc_threshold': 0.4,     # Charging initiation
    'target_soc': 0.8,        # Charging target
}
```

## 📚 Citation

If you use this code or reference our work, please cite:

```bibtex
@inproceedings{kaushik2025decentralized,
  title={Decentralized Charging Coordination for Electric Buses: A Q-Learning Approach},
  author={Kaushik, Diya},
  booktitle={IEEE Conference Proceedings},
  year={2025},
  pages={1--8}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Computing resources provided by Galgotias University
- Faculty members of Department of Computer Science and Engineering
- Dr. Suveg Moudgil for guidance and mentorship

## 📬 Contact

**Diya Kaushik**  
Department of Computer Science and Engineering  
Galgotias University  
Gautam Buddha Nagar, India  
[diyakaushik027@gmail.com](mailto:diyakaushik027@gmail.com)


---

**Related Papers:** [Electric bus charging station placement with queueing considerations](https://doi.org/10.1016/j.trc.2019.01.020) | [Optimal charging scheduling for fast-charging bus systems](https://doi.org/10.1016/j.tre.2019.01.002)

**Tags:** `reinforcement-learning` `q-learning` `electric-buses` `smart-cities` `public-transportation` `charging-coordination`
