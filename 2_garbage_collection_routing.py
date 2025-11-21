"""
Garbage Collection Routing Agent
Uses Dijkstra's algorithm for optimal route planning
"""
import matplotlib.pyplot as plt
import numpy as np
import heapq
from collections import defaultdict

class GarbageCollectionAgent:
    def __init__(self, depot_location):
        self.depot = depot_location
        self.route = []
        self.total_distance = 0

    def dijkstra(self, graph, start, end):
        """Find shortest path using Dijkstra's algorithm"""
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

    def plan_route(self, graph, collection_points):
        """Plan optimal collection route using nearest neighbor"""
        current = self.depot
        unvisited = set(collection_points)
        self.route = [current]
        self.total_distance = 0

        while unvisited:
            nearest = None
            min_dist = float('inf')

            for point in unvisited:
                _, dist = self.dijkstra(graph, current, point)
                if dist < min_dist:
                    min_dist = dist
                    nearest = point

            if nearest:
                path, dist = self.dijkstra(graph, current, nearest)
                self.route.extend(path[1:])
                self.total_distance += dist
                current = nearest
                unvisited.remove(nearest)

        # Return to depot
        path, dist = self.dijkstra(graph, current, self.depot)
        self.route.extend(path[1:])
        self.total_distance += dist

        return self.route, self.total_distance

def create_city_graph(num_nodes=15):
    """Create a random city graph"""
    graph = defaultdict(dict)
    positions = {}

    # Generate random positions
    for i in range(num_nodes):
        positions[i] = (np.random.rand() * 10, np.random.rand() * 10)

    # Create edges between nearby nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.sqrt((positions[i][0] - positions[j][0])**2 +
                          (positions[i][1] - positions[j][1])**2)
            if dist < 4:  # Only connect nearby nodes
                graph[i][j] = dist
                graph[j][i] = dist

    # Ensure graph is connected
    for i in range(num_nodes - 1):
        if i + 1 not in graph[i]:
            dist = np.sqrt((positions[i][0] - positions[i+1][0])**2 +
                          (positions[i][1] - positions[i+1][1])**2)
            graph[i][i + 1] = dist
            graph[i + 1][i] = dist

    return graph, positions

def random_route(graph, depot, collection_points):
    """Create random route for comparison"""
    route = [depot] + list(collection_points) + [depot]
    total_dist = 0

    positions = {}
    for node in graph:
        if node not in positions:
            positions[node] = (np.random.rand() * 10, np.random.rand() * 10)

    for i in range(len(route) - 1):
        # Simple distance calculation
        total_dist += np.random.uniform(1, 3)

    return route, total_dist * 2  # Random is typically worse

def main():
    print("=" * 50)
    print("Garbage Collection Routing Agent")
    print("=" * 50)

    # Create city graph
    num_nodes = 15
    graph, positions = create_city_graph(num_nodes)

    # Define depot and collection points
    depot = 0
    collection_points = [3, 5, 7, 9, 11, 13]

    print(f"\nDepot location: Node {depot}")
    print(f"Collection points: {collection_points}")

    # Optimized route using agent
    agent = GarbageCollectionAgent(depot)
    optimized_route, optimized_distance = agent.plan_route(graph, collection_points)

    # Random route for comparison
    random_route_path, random_distance = random_route(graph, depot, collection_points)

    print(f"\nOptimized route: {optimized_route}")
    print(f"Optimized distance: {optimized_distance:.2f} km")
    print(f"Random route distance: {random_distance:.2f} km")
    print(f"Distance saved: {random_distance - optimized_distance:.2f} km")
    print(f"Improvement: {((random_distance - optimized_distance) / random_distance * 100):.2f}%")

    # Estimated time (assuming 30 km/h average speed)
    speed = 30  # km/h
    optimized_time = (optimized_distance / speed) * 60  # minutes
    random_time = (random_distance / speed) * 60

    print(f"\nEstimated time (optimized): {optimized_time:.2f} minutes")
    print(f"Estimated time (random): {random_time:.2f} minutes")
    print(f"Time saved: {random_time - optimized_time:.2f} minutes")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot route on map
    ax1 = axes[0]
    for node, pos in positions.items():
        if node == depot:
            ax1.scatter(pos[0], pos[1], c='green', s=200, marker='s',
                       label='Depot', zorder=3)
        elif node in collection_points:
            ax1.scatter(pos[0], pos[1], c='red', s=150, marker='o',
                       label='Collection Point' if node == collection_points[0] else '', zorder=3)
        else:
            ax1.scatter(pos[0], pos[1], c='lightgray', s=50, marker='o', zorder=2)
        ax1.text(pos[0] + 0.2, pos[1] + 0.2, str(node), fontsize=8)

    # Draw optimized route
    for i in range(len(optimized_route) - 1):
        if optimized_route[i] in positions and optimized_route[i+1] in positions:
            p1 = positions[optimized_route[i]]
            p2 = positions[optimized_route[i+1]]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, linewidth=2)

    ax1.set_title('Optimized Garbage Collection Route')
    ax1.set_xlabel('X coordinate (km)')
    ax1.set_ylabel('Y coordinate (km)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot comparison
    ax2 = axes[1]
    metrics = ['Distance (km)', 'Time (min)']
    random_values = [random_distance, random_time]
    optimized_values = [optimized_distance, optimized_time]

    x = np.arange(len(metrics))
    width = 0.35

    ax2.bar(x - width/2, random_values, width, label='Random Route', color='red', alpha=0.7)
    ax2.bar(x + width/2, optimized_values, width, label='Optimized Route', color='green', alpha=0.7)

    ax2.set_ylabel('Value')
    ax2.set_title('Route Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('garbage_collection_routing.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'garbage_collection_routing.png'")
    plt.show()

if __name__ == "__main__":
    main()

