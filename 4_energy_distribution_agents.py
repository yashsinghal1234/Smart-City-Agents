"""
Energy Distribution Agent
Balances energy supply among buildings using rule-based logic
"""
import matplotlib.pyplot as plt
import numpy as np

class EnergyDistributionAgent:
    def __init__(self, total_capacity):
        self.total_capacity = total_capacity
        self.buildings = []

    def add_building(self, building_id, demand, priority=1):
        """Add a building with energy demand"""
        self.buildings.append({
            'id': building_id,
            'demand': demand,
            'priority': priority,
            'allocated': 0
        })

    def distribute_energy_balanced(self):
        """Distribute energy using balanced allocation"""
        total_demand = sum(b['demand'] for b in self.buildings)

        if total_demand <= self.total_capacity:
            # Sufficient energy for all
            for building in self.buildings:
                building['allocated'] = building['demand']
        else:
            # Proportional allocation based on priority
            total_priority_demand = sum(b['demand'] * b['priority'] for b in self.buildings)

            for building in self.buildings:
                weighted_demand = building['demand'] * building['priority']
                building['allocated'] = (weighted_demand / total_priority_demand) * self.total_capacity

        return self.buildings

    def distribute_energy_unbalanced(self):
        """Distribute energy using simple first-come-first-serve"""
        remaining_capacity = self.total_capacity

        for building in self.buildings:
            if remaining_capacity >= building['demand']:
                building['allocated'] = building['demand']
                remaining_capacity -= building['demand']
            else:
                building['allocated'] = remaining_capacity
                remaining_capacity = 0

        return self.buildings

    def calculate_balance_efficiency(self):
        """Calculate how balanced the distribution is"""
        if not self.buildings:
            return 0

        allocations = [b['allocated'] / max(b['demand'], 1) for b in self.buildings]
        mean_allocation = np.mean(allocations)
        variance = np.var(allocations)

        # Efficiency is higher when variance is lower
        efficiency = 1 / (1 + variance) * 100
        return efficiency

def simulate_energy_distribution(time_steps=24):
    """Simulate energy distribution over 24 hours"""
    np.random.seed(42)

    balanced_efficiencies = []
    unbalanced_efficiencies = []

    for hour in range(time_steps):
        # Total capacity varies by time of day
        base_capacity = 1000
        capacity = base_capacity + 200 * np.sin(hour * np.pi / 12)

        # Create agent
        agent_balanced = EnergyDistributionAgent(capacity)
        agent_unbalanced = EnergyDistributionAgent(capacity)

        # Add buildings with varying demands
        num_buildings = 10
        for i in range(num_buildings):
            # Demand varies by time and building type
            base_demand = np.random.uniform(80, 150)
            time_factor = 1 + 0.3 * np.sin((hour - 6) * np.pi / 12)
            demand = base_demand * time_factor
            priority = np.random.choice([1, 2, 3])  # 3 is highest priority

            agent_balanced.add_building(i, demand, priority)
            agent_unbalanced.add_building(i, demand, priority)

        # Distribute energy
        agent_balanced.distribute_energy_balanced()
        agent_unbalanced.distribute_energy_unbalanced()

        # Calculate efficiency
        balanced_eff = agent_balanced.calculate_balance_efficiency()
        unbalanced_eff = agent_unbalanced.calculate_balance_efficiency()

        balanced_efficiencies.append(balanced_eff)
        unbalanced_efficiencies.append(unbalanced_eff)

    return balanced_efficiencies, unbalanced_efficiencies

def main():
    print("=" * 50)
    print("Energy Distribution Agent")
    print("=" * 50)

    # Single snapshot example
    print("\nSnapshot at peak hour (6 PM):")
    capacity = 1000
    agent = EnergyDistributionAgent(capacity)

    # Add buildings
    building_demands = [120, 150, 100, 180, 90, 110, 140, 95, 160, 130]
    priorities = [2, 3, 1, 3, 1, 2, 2, 1, 3, 2]

    for i, (demand, priority) in enumerate(zip(building_demands, priorities)):
        agent.add_building(i, demand, priority)

    print(f"Total capacity: {capacity} kW")
    print(f"Total demand: {sum(building_demands)} kW")

    # Balanced distribution
    agent.distribute_energy_balanced()
    balanced_eff = agent.calculate_balance_efficiency()

    print(f"\nBalanced distribution efficiency: {balanced_eff:.2f}%")

    # Simulate over 24 hours
    print("\nSimulating 24-hour energy distribution...")
    balanced_effs, unbalanced_effs = simulate_energy_distribution()

    avg_balanced = np.mean(balanced_effs)
    avg_unbalanced = np.mean(unbalanced_effs)

    print(f"Average balanced efficiency: {avg_balanced:.2f}%")
    print(f"Average unbalanced efficiency: {avg_unbalanced:.2f}%")
    print(f"Improvement: {avg_balanced - avg_unbalanced:.2f}%")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Efficiency over time
    ax1 = axes[0, 0]
    hours = list(range(24))
    ax1.plot(hours, balanced_effs, label='Balanced', color='green',
            linewidth=2, marker='o', markersize=4)
    ax1.plot(hours, unbalanced_effs, label='Unbalanced', color='red',
            linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Load Balance Efficiency (%)')
    ax1.set_title('Energy Distribution Efficiency Over 24 Hours')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 3))

    # Plot 2: Building allocation (snapshot)
    ax2 = axes[0, 1]
    building_ids = [f'B{i}' for i in range(len(agent.buildings))]
    demands = [b['demand'] for b in agent.buildings]
    allocated = [b['allocated'] for b in agent.buildings]

    x = np.arange(len(building_ids))
    width = 0.35

    ax2.bar(x - width/2, demands, width, label='Demand', color='orange', alpha=0.7)
    ax2.bar(x + width/2, allocated, width, label='Allocated', color='green', alpha=0.7)
    ax2.set_xlabel('Building')
    ax2.set_ylabel('Energy (kW)')
    ax2.set_title('Energy Demand vs Allocation (Peak Hour)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(building_ids, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Average efficiency comparison
    ax3 = axes[1, 0]
    methods = ['Balanced\nDistribution', 'Unbalanced\nDistribution']
    efficiencies = [avg_balanced, avg_unbalanced]
    colors = ['green', 'red']

    bars = ax3.bar(methods, efficiencies, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Efficiency (%)')
    ax3.set_title('Distribution Method Comparison')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}%', ha='center', va='bottom')

    # Plot 4: Priority-based allocation
    ax4 = axes[1, 1]
    priority_levels = [1, 2, 3]
    priority_colors = ['yellow', 'orange', 'red']

    for i, building in enumerate(agent.buildings):
        priority = building['priority']
        allocation_ratio = building['allocated'] / building['demand'] if building['demand'] > 0 else 0
        ax4.scatter(priority, allocation_ratio,
                   c=priority_colors[priority-1], s=150, alpha=0.6, edgecolors='black')

    ax4.set_xlabel('Priority Level (1=Low, 3=High)')
    ax4.set_ylabel('Allocation Ratio (Allocated/Demand)')
    ax4.set_title('Priority vs Allocation Ratio')
    ax4.set_xticks(priority_levels)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('energy_distribution.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'energy_distribution.png'")
    plt.show()

if __name__ == "__main__":
    main()

