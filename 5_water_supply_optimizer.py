"""
Water Supply Optimizer Agent
Manages water distribution using constraint satisfaction
"""
import matplotlib.pyplot as plt
import numpy as np

class WaterSupplyAgent:
    def __init__(self, total_capacity, min_pressure=2.0, max_pressure=5.0):
        self.total_capacity = total_capacity  # liters per minute
        self.min_pressure = min_pressure  # bar
        self.max_pressure = max_pressure  # bar
        self.houses = []

    def add_house(self, house_id, demand, distance, elevation):
        """Add a house with water demand"""
        self.houses.append({
            'id': house_id,
            'demand': demand,  # liters per minute
            'distance': distance,  # meters from source
            'elevation': elevation,  # meters above source
            'flow': 0,
            'pressure': 0
        })

    def calculate_pressure(self, flow, distance, elevation):
        """Calculate pressure at house based on flow and distance"""
        # Simplified pressure calculation
        # Pressure loss due to distance and elevation
        base_pressure = 4.0  # bar
        distance_loss = distance * 0.01  # 0.01 bar per meter
        elevation_loss = elevation * 0.1  # 0.1 bar per meter elevation
        flow_loss = flow * 0.002  # pressure decreases with high flow

        pressure = base_pressure - distance_loss - elevation_loss - flow_loss
        return max(pressure, 0)

    def optimize_distribution(self):
        """Optimize water distribution with constraints"""
        total_demand = sum(h['demand'] for h in self.houses)

        # If we have enough capacity, try to meet all demands
        if total_demand <= self.total_capacity:
            for house in self.houses:
                house['flow'] = house['demand']
        else:
            # Proportional distribution based on demand and priority
            # Priority: houses with higher elevation and closer distance get preference
            for house in self.houses:
                priority_factor = 1 / (1 + house['elevation'] * 0.1 + house['distance'] * 0.001)
                house['flow'] = (house['demand'] / total_demand) * self.total_capacity * priority_factor

        # Calculate pressures and adjust flow to meet pressure constraints
        for house in self.houses:
            pressure = self.calculate_pressure(house['flow'], house['distance'], house['elevation'])

            # Adjust flow if pressure is outside bounds
            if pressure < self.min_pressure:
                # Increase flow might not help, but we ensure minimum
                house['flow'] = max(house['flow'], house['demand'] * 0.7)
                pressure = self.calculate_pressure(house['flow'], house['distance'], house['elevation'])
            elif pressure > self.max_pressure:
                # Reduce flow to maintain pressure
                house['flow'] = house['flow'] * 0.8
                pressure = self.calculate_pressure(house['flow'], house['distance'], house['elevation'])

            house['pressure'] = pressure

        return self.houses

    def calculate_satisfaction(self):
        """Calculate overall satisfaction metric"""
        if not self.houses:
            return 0

        satisfaction_scores = []
        for house in self.houses:
            # Satisfaction based on meeting demand and pressure constraints
            demand_satisfaction = house['flow'] / house['demand'] if house['demand'] > 0 else 1
            pressure_in_range = (self.min_pressure <= house['pressure'] <= self.max_pressure)
            pressure_satisfaction = 1.0 if pressure_in_range else 0.5

            satisfaction = demand_satisfaction * pressure_satisfaction
            satisfaction_scores.append(min(satisfaction, 1.0))

        return np.mean(satisfaction_scores) * 100

def main():
    print("=" * 50)
    print("Water Supply Optimizer Agent")
    print("=" * 50)

    # Create water supply system
    total_capacity = 500  # liters per minute
    agent = WaterSupplyAgent(total_capacity)

    # Add houses with different characteristics
    houses_data = [
        (1, 50, 100, 0),    # house_id, demand, distance, elevation
        (2, 60, 150, 2),
        (3, 45, 80, 1),
        (4, 70, 200, 5),
        (5, 55, 120, 3),
        (6, 40, 90, 1),
        (7, 65, 180, 4),
        (8, 50, 110, 2),
        (9, 48, 140, 3),
        (10, 52, 95, 1)
    ]

    print(f"\nTotal system capacity: {total_capacity} L/min")
    print(f"Number of houses: {len(houses_data)}")

    for house_data in houses_data:
        agent.add_house(*house_data)

    total_demand = sum(h[1] for h in houses_data)
    print(f"Total demand: {total_demand} L/min")

    # Optimize distribution
    print("\nOptimizing water distribution...")
    optimized = agent.optimize_distribution()
    satisfaction = agent.calculate_satisfaction()

    print(f"Overall satisfaction: {satisfaction:.2f}%")

    # Print house details
    print("\nHouse-wise distribution:")
    print(f"{'House':<8} {'Demand':<10} {'Flow':<10} {'Pressure':<10} {'Status':<10}")
    print("-" * 55)

    for house in optimized:
        status = "OK" if agent.min_pressure <= house['pressure'] <= agent.max_pressure else "WARNING"
        print(f"H{house['id']:<7} {house['demand']:<10.1f} {house['flow']:<10.1f} "
              f"{house['pressure']:<10.2f} {status:<10}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Flow distribution
    ax1 = axes[0, 0]
    house_ids = [f"H{h['id']}" for h in optimized]
    demands = [h['demand'] for h in optimized]
    flows = [h['flow'] for h in optimized]

    x = np.arange(len(house_ids))
    width = 0.35

    ax1.bar(x - width/2, demands, width, label='Demand', color='orange', alpha=0.7)
    ax1.bar(x + width/2, flows, width, label='Allocated Flow', color='blue', alpha=0.7)
    ax1.set_xlabel('House')
    ax1.set_ylabel('Flow Rate (L/min)')
    ax1.set_title('Water Flow Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(house_ids)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Pressure distribution
    ax2 = axes[0, 1]
    pressures = [h['pressure'] for h in optimized]
    colors = ['green' if agent.min_pressure <= p <= agent.max_pressure else 'red'
              for p in pressures]

    bars = ax2.bar(house_ids, pressures, color=colors, alpha=0.7)
    ax2.axhline(y=agent.min_pressure, color='orange', linestyle='--',
                label=f'Min Pressure ({agent.min_pressure} bar)')
    ax2.axhline(y=agent.max_pressure, color='red', linestyle='--',
                label=f'Max Pressure ({agent.max_pressure} bar)')
    ax2.set_xlabel('House')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Water Pressure Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Network topology
    ax3 = axes[1, 0]
    distances = [h['distance'] for h in optimized]
    elevations = [h['elevation'] for h in optimized]

    # Source at origin
    ax3.scatter(0, 0, c='blue', s=300, marker='s', label='Water Source', zorder=3)

    # Houses
    scatter = ax3.scatter(distances, elevations, c=pressures, s=200,
                         cmap='RdYlGn', alpha=0.7, edgecolors='black',
                         label='Houses', zorder=2)

    # Connect houses to source
    for dist, elev in zip(distances, elevations):
        ax3.plot([0, dist], [0, elev], 'gray', alpha=0.3, linestyle='--')

    # Labels
    for i, (dist, elev) in enumerate(zip(distances, elevations)):
        ax3.text(dist, elev, f'H{i+1}', fontsize=8, ha='center', va='center')

    ax3.set_xlabel('Distance from Source (m)')
    ax3.set_ylabel('Elevation (m)')
    ax3.set_title('Water Distribution Network')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Pressure (bar)')

    # Plot 4: Satisfaction metrics
    ax4 = axes[1, 1]

    # Calculate individual satisfaction scores
    satisfaction_scores = []
    for house in optimized:
        demand_sat = (house['flow'] / house['demand']) * 100 if house['demand'] > 0 else 100
        pressure_ok = agent.min_pressure <= house['pressure'] <= agent.max_pressure
        total_sat = demand_sat if pressure_ok else demand_sat * 0.5
        satisfaction_scores.append(min(total_sat, 100))

    bars = ax4.barh(house_ids, satisfaction_scores, color='green', alpha=0.7)
    ax4.axvline(x=80, color='orange', linestyle='--', label='Target (80%)')
    ax4.set_xlabel('Satisfaction Score (%)')
    ax4.set_ylabel('House')
    ax4.set_title('House-wise Water Supply Satisfaction')
    ax4.set_xlim([0, 105])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('graphs/water_supply_optimization.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'graphs/water_supply_optimization.png'")
    plt.show()

if __name__ == "__main__":
    main()
