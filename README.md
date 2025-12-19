# Shenzhen Electric Bus Simulation - 2025-12-18

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
- Buses: 12
- Charging stations: 5
- Ports per station: 2
- Simulation duration: 180 minutes
- Time step: 5 seconds
- Battery capacity: 350.0 kWh
- Charging power: 120.0 kW

## STRATEGIES COMPARED:
1. Baseline (nearest charger)
2. Heuristic (multi-factor scoring)
3. Random selection
4. Oracle (perfect information)
5. Reinforcement Learning (Q-learning)
