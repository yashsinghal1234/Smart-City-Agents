# 🏙️ Smart City Multi-Agent Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

A comprehensive collection of 10 AI-powered autonomous agents designed to optimize various smart city operations. Each agent uses proven optimization algorithms and machine learning techniques to solve real-world urban challenges.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Agents](#agents)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Technologies Used](#technologies-used)
- [Output Examples](#output-examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements intelligent agents that can autonomously optimize critical smart city infrastructure and services. Each agent demonstrates practical applications of AI/ML algorithms in solving urban management challenges, from traffic optimization to waste management.

**Key Highlights:**
- ✅ 10 fully functional autonomous agents
- ✅ Simple, clean, and well-documented code
- ✅ Real-time visualization of results
- ✅ Measurable performance improvements
- ✅ Production-ready implementations

## ✨ Features

- **Autonomous Decision Making**: Agents make real-time decisions without human intervention
- **Optimization Algorithms**: Implementation of Dijkstra, Hungarian Algorithm, Reinforcement Learning, etc.
- **Machine Learning**: Scikit-learn based classification for intelligent waste sorting
- **Visual Analytics**: Comprehensive graphs and heatmaps for performance analysis
- **Scalable Architecture**: Easy to extend and modify for different city sizes
- **Comparative Analysis**: Before/after comparisons showing clear improvements

## 🤖 Agents

### 1. 🚦 Traffic Light Optimizer
**File:** `1_traffic_light_optimizer.py`

Optimizes traffic signal timing to reduce congestion and waiting times at intersections.

- **Algorithm**: Reinforcement learning-based adaptive timing
- **Input**: Real-time traffic density at intersections
- **Output**: Optimized green light durations
- **Improvement**: 20-30% reduction in average delay

**Use Case**: Reduces traffic congestion during peak hours by dynamically adjusting signal timings based on traffic flow.

---

### 2. 🗑️ Garbage Collection Routing
**File:** `2_garbage_collection_routing.py`

Plans optimal routes for garbage collection trucks to minimize travel distance and time.

- **Algorithm**: Dijkstra's shortest path + Nearest Neighbor
- **Input**: Depot location, collection points, city graph
- **Output**: Optimized collection route
- **Improvement**: 25-35% reduction in travel distance

**Use Case**: Reduces fuel costs and collection time by finding the most efficient route through all collection points.

---

### 3. 🅿️ Smart Parking Allocation
**File:** `3_smart_parking_allocation.py`

Allocates available parking spots to vehicles optimally to minimize walking distance.

- **Algorithm**: Hungarian Algorithm for optimal assignment
- **Input**: Vehicle locations, available parking spots
- **Output**: Parking usage heatmap
- **Improvement**: 15-25% reduction in average walking distance

**Use Case**: Helps drivers find the nearest available parking spot, reducing search time and emissions.

---

### 4. ⚡ Energy Distribution Agents
**File:** `4_energy_distribution_agents.py`

Balances electrical energy supply across buildings based on demand and priority.

- **Algorithm**: Priority-based rule logic with constraint satisfaction
- **Input**: Total capacity, building demands, priorities
- **Output**: Load balance efficiency over 24 hours
- **Improvement**: 40-60% better load balancing efficiency

**Use Case**: Prevents power outages and ensures critical infrastructure gets priority during high demand.

---

### 5. 💧 Water Supply Optimizer
**File:** `5_water_supply_optimizer.py`

Manages water distribution ensuring adequate pressure and flow to all houses.

- **Algorithm**: Constraint satisfaction with pressure calculations
- **Input**: Houses with demand, distance, elevation
- **Output**: Water flow and pressure distribution
- **Improvement**: 80-90% satisfaction rate maintained

**Use Case**: Ensures all households receive adequate water supply while maintaining safe pressure levels.

---

### 6. 🚑 Emergency Response Dispatchers
**File:** `6_emergency_response_dispatchers.py`

Assigns emergency vehicles (ambulances, fire trucks) to incidents optimally.

- **Algorithm**: Nearest-neighbor with severity-based priority
- **Input**: Ambulance locations, incident locations and severity
- **Output**: Average response time analysis
- **Improvement**: 30-40% faster response times

**Use Case**: Saves lives by ensuring fastest possible response to emergency situations.

---

### 7. 🌫️ Pollution Control Monitors
**File:** `7_pollution_control_monitors.py`

Monitors air pollution levels and coordinates mitigation efforts.

- **Algorithm**: Grid-based simulation with distributed alerts
- **Input**: Pollution sources, diffusion model
- **Output**: Pollution levels over time
- **Improvement**: 35-45% reduction in pollution levels

**Use Case**: Identifies pollution hotspots and triggers mitigation measures to improve air quality.

---

### 8. 💡 Streetlight Energy Saver
**File:** `8_streetlight_energy_saver.py`

Adaptively controls streetlight brightness based on motion detection.

- **Algorithm**: Motion sensor-based adaptive dimming
- **Input**: Pedestrian movement patterns
- **Output**: Total energy consumption and savings
- **Improvement**: 40-60% energy savings

**Use Case**: Reduces electricity costs while maintaining public safety through smart lighting.

---

### 9. 🚌 Smart Bus Routing
**File:** `9_smart_bus_routing.py`

Optimizes public bus routes to minimize passenger wait times.

- **Algorithm**: Dijkstra's algorithm on city graph
- **Input**: Bus stops, city network
- **Output**: Optimized route, wait time reduction
- **Improvement**: 20-35% reduction in passenger wait times

**Use Case**: Improves public transportation efficiency and passenger satisfaction.

---

### 10. ♻️ Waste Segregation AI
**File:** `10_waste_segregation_ai.py`

Automatically classifies waste into categories for proper recycling.

- **Algorithm**: Random Forest Classifier (scikit-learn)
- **Input**: Waste item features (weight, size, density, etc.)
- **Output**: Classification accuracy, confusion matrix
- **Improvement**: 85-95% classification accuracy

**Use Case**: Automates waste sorting to improve recycling rates and reduce landfill waste.

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yashsinghal1234/Smart-City-Agents.git
cd Smart-City-Agents
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies Include:
- `matplotlib` - Data visualization
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning
- `scipy` - Scientific computing
- `networkx` - Graph algorithms

## 💻 Usage

### Running Individual Agents

Each agent can be executed independently:

```bash
python 1_traffic_light_optimizer.py
```

Replace the number with any agent (1-10) you want to run.

### Running All Agents

Execute all agents sequentially with a summary report:

```bash
python run_all_agents.py
```

This will:
1. Run all 10 agents one by one
2. Display console output for each
3. Generate visualization graphs
4. Provide a summary report with timing statistics

### Expected Output

Each agent produces:
1. **Console Output**: Detailed metrics and improvement statistics
2. **Visualization**: PNG graphs showing before/after comparisons
3. **Performance Metrics**: Quantifiable improvements

## 📁 Project Structure

```
Smart-City-Agents/
│
├── 1_traffic_light_optimizer.py          # Traffic signal optimization
├── 2_garbage_collection_routing.py       # Waste collection routing
├── 3_smart_parking_allocation.py         # Parking spot assignment
├── 4_energy_distribution_agents.py       # Energy load balancing
├── 5_water_supply_optimizer.py           # Water distribution management
├── 6_emergency_response_dispatchers.py   # Emergency vehicle dispatch
├── 7_pollution_control_monitors.py       # Air quality monitoring
├── 8_streetlight_energy_saver.py         # Adaptive street lighting
├── 9_smart_bus_routing.py                # Public transport optimization
├── 10_waste_segregation_ai.py            # AI-based waste classification
│
├── run_all_agents.py                     # Batch execution script
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
│
└── outputs/                              # Generated visualizations
    ├── traffic_light_optimization.png
    ├── garbage_collection_routing.png
    ├── smart_parking_allocation.png
    ├── energy_distribution.png
    ├── water_supply_optimization.png
    ├── emergency_response.png
    ├── pollution_control.png
    ├── streetlight_energy_saver.png
    ├── smart_bus_routing.png
    └── waste_segregation.png
```

## 📊 Performance Metrics

| Agent | Metric | Improvement |
|-------|--------|-------------|
| Traffic Light | Average Delay | ↓ 20-30% |
| Garbage Collection | Travel Distance | ↓ 25-35% |
| Smart Parking | Walking Distance | ↓ 15-25% |
| Energy Distribution | Load Balance Efficiency | ↑ 40-60% |
| Water Supply | Satisfaction Rate | 80-90% |
| Emergency Response | Response Time | ↓ 30-40% |
| Pollution Control | Pollution Levels | ↓ 35-45% |
| Streetlight | Energy Consumption | ↓ 40-60% |
| Bus Routing | Passenger Wait Time | ↓ 20-35% |
| Waste Segregation | Classification Accuracy | 85-95% |

## 🛠️ Technologies Used

### Algorithms
- **Dijkstra's Algorithm** - Shortest path finding
- **Hungarian Algorithm** - Optimal assignment problem
- **Reinforcement Learning** - Adaptive decision making
- **Random Forest** - Machine learning classification
- **Greedy Algorithms** - Optimization heuristics
- **Constraint Satisfaction** - Resource allocation

### Libraries
- **NumPy** - Array operations and mathematical functions
- **Matplotlib** - Data visualization and plotting
- **Scikit-learn** - Machine learning models
- **SciPy** - Optimization and scientific computing
- **NetworkX** - Graph theory and network analysis

### Programming Concepts
- Object-Oriented Programming (OOP)
- Multi-agent systems
- Simulation and modeling
- Real-time optimization
- Data-driven decision making

## 📸 Output Examples

### Sample Visualization - Traffic Light Optimizer
```
Average delay (baseline): 245.32 car-minutes
Average delay (optimized): 178.45 car-minutes
Improvement: 27.26%
```

### Sample Visualization - Energy Distribution
```
Total capacity: 1000 kW
Total demand: 1475 kW
Balanced distribution efficiency: 87.34%
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution
- Adding new agent types
- Improving optimization algorithms
- Enhanced visualization
- Real-world data integration
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Yash Singhal**
- GitHub: [@yashsinghal1234](https://github.com/yashsinghal1234)
- Repository: [Smart-City-Agents](https://github.com/yashsinghal1234/Smart-City-Agents)

## 🙏 Acknowledgments

- Inspired by real-world smart city initiatives
- Built for educational and demonstration purposes
- Algorithms based on established computer science principles

## 🎓 Educational Use

This project is ideal for:
- Computer Science students learning AI/ML
- Hackathons and coding competitions
- Research in smart city technologies
- Portfolio projects
- Teaching optimization algorithms

## 📝 Citation

If you use this project in your research or work, please cite:

```
@software{smart_city_agents,
  author = {Singhal, Yash},
  title = {Smart City Multi-Agent Systems},
  year = {2025},
  url = {https://github.com/yashsinghal1234/Smart-City-Agents}
}
```

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for building smarter cities

[Report Bug](https://github.com/yashsinghal1234/Smart-City-Agents/issues) · [Request Feature](https://github.com/yashsinghal1234/Smart-City-Agents/issues)

</div>

