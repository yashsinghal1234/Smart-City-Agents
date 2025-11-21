"""
Traffic Light Optimizer Agent
Uses reinforcement learning approach to optimize traffic flow
"""
import random
import matplotlib.pyplot as plt
import numpy as np

class TrafficLightAgent:
    def __init__(self, intersection_id, initial_green_time=30):
        self.id = intersection_id
        self.green_time = initial_green_time
        self.cars_waiting = 0
        self.total_wait_time = 0

    def update_waiting_cars(self, num_cars):
        """Update number of cars waiting"""
        self.cars_waiting = num_cars

    def calculate_delay(self):
        """Calculate delay based on cars waiting"""
        return self.cars_waiting * (60 - self.green_time) / 60

    def optimize_timing(self, traffic_density):
        """Adjust green light timing based on traffic density"""
        if traffic_density > 0.7:
            self.green_time = min(60, self.green_time + 5)
        elif traffic_density < 0.3:
            self.green_time = max(15, self.green_time - 5)
        return self.green_time

def simulate_traffic(num_intersections=4, time_steps=100, optimized=False):
    """Simulate traffic flow through intersections"""
    agents = [TrafficLightAgent(i) for i in range(num_intersections)]
    delays = []

    for step in range(time_steps):
        total_delay = 0

        for agent in agents:
            # Simulate random traffic
            cars = random.randint(5, 30)
            agent.update_waiting_cars(cars)

            if optimized:
                # Calculate traffic density
                density = cars / 30
                agent.optimize_timing(density)

            delay = agent.calculate_delay()
            total_delay += delay

        delays.append(total_delay / num_intersections)

    return delays

def main():
    print("=" * 50)
    print("Traffic Light Optimizer Agent")
    print("=" * 50)

    # Run simulation without optimization
    print("\nRunning baseline simulation...")
    baseline_delays = simulate_traffic(optimized=False)

    # Run simulation with optimization
    print("Running optimized simulation...")
    optimized_delays = simulate_traffic(optimized=True)

    # Calculate statistics
    avg_baseline = np.mean(baseline_delays)
    avg_optimized = np.mean(optimized_delays)
    improvement = ((avg_baseline - avg_optimized) / avg_baseline) * 100

    print(f"\nResults:")
    print(f"Average delay (baseline): {avg_baseline:.2f} car-minutes")
    print(f"Average delay (optimized): {avg_optimized:.2f} car-minutes")
    print(f"Improvement: {improvement:.2f}%")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(baseline_delays, label='Baseline', color='red', alpha=0.7)
    plt.plot(optimized_delays, label='Optimized', color='green', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Average Delay (car-minutes)')
    plt.title('Traffic Light Optimization Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(['Baseline', 'Optimized'], [avg_baseline, avg_optimized],
            color=['red', 'green'], alpha=0.7)
    plt.ylabel('Average Delay (car-minutes)')
    plt.title('Average Delay Comparison')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('graphs/traffic_light_optimization.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'graphs/traffic_light_optimization.png'")
    plt.show()

if __name__ == "__main__":
    main()
