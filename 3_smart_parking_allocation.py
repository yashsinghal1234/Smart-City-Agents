"""
Smart Parking Allocation Agent
Uses Hungarian algorithm for optimal parking assignment
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

class ParkingAgent:
    def __init__(self, num_spots, num_zones):
        self.num_spots = num_spots
        self.num_zones = num_zones
        self.parking_grid = np.zeros((num_zones, num_spots // num_zones))

    def calculate_cost_matrix(self, vehicles, spots):
        """Calculate cost matrix based on distance"""
        cost_matrix = np.zeros((len(vehicles), len(spots)))

        for i, vehicle in enumerate(vehicles):
            for j, spot in enumerate(spots):
                # Cost is distance between vehicle and spot
                distance = np.sqrt((vehicle[0] - spot[0])**2 +
                                 (vehicle[1] - spot[1])**2)
                cost_matrix[i][j] = distance

        return cost_matrix

    def allocate_parking(self, vehicles, spots):
        """Allocate parking using Hungarian algorithm"""
        cost_matrix = self.calculate_cost_matrix(vehicles, spots)

        # Use Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        total_cost = cost_matrix[row_ind, col_ind].sum()
        assignments = list(zip(row_ind, col_ind))

        return assignments, total_cost

    def greedy_allocation(self, vehicles, spots):
        """Greedy allocation for comparison"""
        assignments = []
        total_cost = 0
        used_spots = set()

        for i, vehicle in enumerate(vehicles):
            min_dist = float('inf')
            best_spot = None

            for j, spot in enumerate(spots):
                if j not in used_spots:
                    distance = np.sqrt((vehicle[0] - spot[0])**2 +
                                     (vehicle[1] - spot[1])**2)
                    if distance < min_dist:
                        min_dist = distance
                        best_spot = j

            if best_spot is not None:
                assignments.append((i, best_spot))
                total_cost += min_dist
                used_spots.add(best_spot)

        return assignments, total_cost

    def update_parking_grid(self, assignments, spots):
        """Update parking grid based on assignments"""
        self.parking_grid.fill(0)

        for _, spot_idx in assignments:
            spot = spots[spot_idx]
            # Map spot to grid position
            zone = int(spot[1] * self.num_zones / 10)
            slot = int(spot[0] * (self.num_spots // self.num_zones) / 10)
            zone = min(zone, self.num_zones - 1)
            slot = min(slot, self.num_spots // self.num_zones - 1)
            self.parking_grid[zone][slot] += 1

def main():
    print("=" * 50)
    print("Smart Parking Allocation Agent")
    print("=" * 50)

    # Generate random vehicles and parking spots
    num_vehicles = 25
    num_spots = 30
    num_zones = 5

    np.random.seed(42)
    vehicles = [(np.random.rand() * 10, np.random.rand() * 10)
                for _ in range(num_vehicles)]
    spots = [(np.random.rand() * 10, np.random.rand() * 10)
             for _ in range(num_spots)]

    print(f"\nNumber of vehicles: {num_vehicles}")
    print(f"Number of available spots: {num_spots}")

    # Create agent
    agent = ParkingAgent(num_spots, num_zones)

    # Optimal allocation using Hungarian algorithm
    print("\nCalculating optimal allocation...")
    optimal_assignments, optimal_cost = agent.allocate_parking(vehicles, spots)

    # Greedy allocation for comparison
    print("Calculating greedy allocation...")
    greedy_assignments, greedy_cost = agent.greedy_allocation(vehicles, spots)

    # Update parking grid
    agent.update_parking_grid(optimal_assignments, spots)

    print(f"\nOptimal total distance: {optimal_cost:.2f} units")
    print(f"Greedy total distance: {greedy_cost:.2f} units")
    print(f"Distance saved: {greedy_cost - optimal_cost:.2f} units")
    print(f"Improvement: {((greedy_cost - optimal_cost) / greedy_cost * 100):.2f}%")

    avg_optimal = optimal_cost / len(optimal_assignments)
    avg_greedy = greedy_cost / len(greedy_assignments)
    print(f"\nAverage distance per vehicle (optimal): {avg_optimal:.2f} units")
    print(f"Average distance per vehicle (greedy): {avg_greedy:.2f} units")

    # Plotting
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Optimal allocation map
    ax1 = fig.add_subplot(131)
    vehicle_x = [v[0] for v in vehicles]
    vehicle_y = [v[1] for v in vehicles]
    spot_x = [s[0] for s in spots]
    spot_y = [s[1] for s in spots]

    ax1.scatter(vehicle_x, vehicle_y, c='blue', s=100, marker='o',
               label='Vehicles', alpha=0.6)
    ax1.scatter(spot_x, spot_y, c='green', s=100, marker='s',
               label='Parking Spots', alpha=0.6)

    # Draw assignment lines
    for vehicle_idx, spot_idx in optimal_assignments:
        v = vehicles[vehicle_idx]
        s = spots[spot_idx]
        ax1.plot([v[0], s[0]], [v[1], s[1]], 'r-', alpha=0.3, linewidth=1)

    ax1.set_title('Optimal Parking Allocation')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parking usage heatmap
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(agent.parking_grid, cmap='YlOrRd', aspect='auto')
    ax2.set_title('Parking Usage Heatmap')
    ax2.set_xlabel('Parking Slot')
    ax2.set_ylabel('Zone')
    plt.colorbar(im, ax=ax2, label='Occupancy')

    # Plot 3: Cost comparison
    ax3 = fig.add_subplot(133)
    methods = ['Optimal\n(Hungarian)', 'Greedy']
    costs = [optimal_cost, greedy_cost]
    colors = ['green', 'orange']

    bars = ax3.bar(methods, costs, color=colors, alpha=0.7)
    ax3.set_ylabel('Total Distance (units)')
    ax3.set_title('Allocation Method Comparison')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('smart_parking_allocation.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'smart_parking_allocation.png'")
    plt.show()

if __name__ == "__main__":
    main()

