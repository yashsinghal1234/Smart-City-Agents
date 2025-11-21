"""
Smart Bus Routing Agent
Optimizes bus routes using graph-based algorithms
"""
import matplotlib.pyplot as plt
import numpy as np
import heapq
from collections import defaultdict

class BusRoutingAgent:
    def __init__(self, bus_id):
        self.bus_id = bus_id
        self.route = []
        self.total_distance = 0

    def dijkstra_path(self, graph, start, end):
        """Find shortest path using Dijkstra"""
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        pq = [(0, start)]
        previous = {}

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current == end:
                break

            if current_dist > distances[current]:
                continue

            for neighbor, weight in graph[current].items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))

        # Reconstruct path
        path = []
        current = end
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()

        return path, distances[end]

    def plan_route(self, graph, stops):
        """Plan optimal route through all stops"""
        if len(stops) < 2:
            return [], 0

        self.route = []
        self.total_distance = 0

        for i in range(len(stops) - 1):
            path, distance = self.dijkstra_path(graph, stops[i], stops[i+1])
            if i == 0:
                self.route.extend(path)
            else:
                self.route.extend(path[1:])
            self.total_distance += distance

        return self.route, self.total_distance

def create_city_network():
    """Create a city street network"""
    graph = defaultdict(dict)
    positions = {}

    # Create grid network
    grid_size = 6
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            positions[node_id] = (j * 2, i * 2)

            # Connect to right neighbor
            if j < grid_size - 1:
                right = i * grid_size + (j + 1)
                dist = 2 + np.random.uniform(-0.3, 0.3)
                graph[node_id][right] = dist
                graph[right][node_id] = dist

            # Connect to bottom neighbor
            if i < grid_size - 1:
                bottom = (i + 1) * grid_size + j
                dist = 2 + np.random.uniform(-0.3, 0.3)
                graph[node_id][bottom] = dist
                graph[bottom][node_id] = dist

    return graph, positions

def calculate_passenger_wait_time(route_length, num_passengers, avg_speed=30):
    """Calculate average passenger wait time"""
    # Route time in minutes
    route_time = (route_length / avg_speed) * 60
    # Average wait time is half the route cycle time
    avg_wait = route_time / 2
    return avg_wait

def main():
    print("=" * 50)
    print("Smart Bus Routing Agent")
    print("=" * 50)

    # Create city network
    graph, positions = create_city_network()

    # Define bus stops
    bus_stops = [0, 5, 11, 17, 23, 29, 35]  # Key locations

    print(f"\nNumber of bus stops: {len(bus_stops)}")

    # Create agent and plan optimal route
    agent = BusRoutingAgent(bus_id=1)
    optimal_route, optimal_distance = agent.plan_route(graph, bus_stops)

    # Compare with direct route (without optimization)
    direct_distance = len(bus_stops) * 3  # Rough estimate

    print(f"Optimal route length: {optimal_distance:.2f} km")
    print(f"Direct route estimate: {direct_distance:.2f} km")

    # Calculate wait times
    num_passengers = 150
    optimal_wait = calculate_passenger_wait_time(optimal_distance, num_passengers)
    direct_wait = calculate_passenger_wait_time(direct_distance * 1.5, num_passengers)

    wait_reduction = direct_wait - optimal_wait
    reduction_percent = (wait_reduction / direct_wait) * 100

    print(f"\nAverage passenger wait time (optimal): {optimal_wait:.2f} minutes")
    print(f"Average passenger wait time (direct): {direct_wait:.2f} minutes")
    print(f"Wait time reduction: {wait_reduction:.2f} minutes ({reduction_percent:.2f}%)")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Route map
    ax1 = axes[0]

    # Draw network edges
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if node < neighbor:  # Draw each edge once
                x = [positions[node][0], positions[neighbor][0]]
                y = [positions[node][1], positions[neighbor][1]]
                ax1.plot(x, y, 'gray', alpha=0.3, linewidth=1)

    # Draw all nodes
    for node, pos in positions.items():
        ax1.scatter(pos[0], pos[1], c='lightgray', s=50, zorder=2)

    # Highlight bus stops
    for stop in bus_stops:
        pos = positions[stop]
        ax1.scatter(pos[0], pos[1], c='red', s=200, marker='s',
                   zorder=3, edgecolors='black', linewidth=2)
        ax1.text(pos[0], pos[1]-0.4, f'S{bus_stops.index(stop)}',
                ha='center', fontsize=9, fontweight='bold')

    # Draw optimal route
    for i in range(len(optimal_route) - 1):
        if optimal_route[i] in positions and optimal_route[i+1] in positions:
            x = [positions[optimal_route[i]][0], positions[optimal_route[i+1]][0]]
            y = [positions[optimal_route[i]][1], positions[optimal_route[i+1]][1]]
            ax1.plot(x, y, 'blue', alpha=0.7, linewidth=3, zorder=1)

    ax1.set_title('Optimized Bus Route')
    ax1.set_xlabel('X coordinate (km)')
    ax1.set_ylabel('Y coordinate (km)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Network', 'Bus Stops', 'Optimal Route'])

    # Plot 2: Wait time comparison
    ax2 = axes[1]

    methods = ['Optimized\nRoute', 'Direct\nRoute']
    wait_times = [optimal_wait, direct_wait]
    colors = ['green', 'orange']

    bars = ax2.bar(methods, wait_times, color=colors, alpha=0.7)
    ax2.set_ylabel('Average Wait Time (minutes)')
    ax2.set_title('Passenger Wait Time Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, wait in zip(bars, wait_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{wait:.1f} min', ha='center', va='bottom')

    # Add improvement text
    ax2.text(0.5, max(wait_times) * 0.5,
            f'Reduction:\n{wait_reduction:.1f} min\n({reduction_percent:.1f}%)',
            ha='center', fontsize=12, bbox=dict(boxstyle='round',
            facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig('smart_bus_routing.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'smart_bus_routing.png'")
    plt.show()

if __name__ == "__main__":
    main()

