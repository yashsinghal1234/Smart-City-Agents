"""
Pollution Control Monitor Agent
Monitors pollution and coordinates mitigation using grid simulation
"""
import matplotlib.pyplot as plt
import numpy as np

class PollutionMonitorAgent:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.pollution_grid = np.zeros((grid_size, grid_size))
        self.alert_threshold = 50
        self.critical_threshold = 80

    def add_pollution_source(self, x, y, intensity):
        """Add pollution at location"""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.pollution_grid[y][x] += intensity

    def diffuse_pollution(self):
        """Simulate pollution diffusion"""
        new_grid = self.pollution_grid.copy()

        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                # Average with neighbors
                neighbors = [
                    self.pollution_grid[i-1][j],
                    self.pollution_grid[i+1][j],
                    self.pollution_grid[i][j-1],
                    self.pollution_grid[i][j+1]
                ]
                new_grid[i][j] = (self.pollution_grid[i][j] * 0.6 +
                                 np.mean(neighbors) * 0.4)

        self.pollution_grid = new_grid
        return new_grid

    def apply_mitigation(self, zones):
        """Apply pollution mitigation in specified zones"""
        for x, y in zones:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Reduce pollution by 30%
                self.pollution_grid[y][x] *= 0.7

                # Also reduce in nearby cells
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.pollution_grid[ny][nx] *= 0.85

    def detect_alerts(self):
        """Detect areas with high pollution"""
        alerts = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.pollution_grid[i][j] > self.alert_threshold:
                    alerts.append((j, i, self.pollution_grid[i][j]))
        return alerts

    def get_average_pollution(self):
        """Get average pollution level"""
        return np.mean(self.pollution_grid)

def simulate_pollution_control(time_steps=50):
    """Simulate pollution control over time"""
    agent = PollutionMonitorAgent(grid_size=10)

    # Pollution sources (factories, traffic)
    sources = [(2, 2), (7, 3), (5, 8), (8, 7)]

    avg_pollution_no_control = []
    avg_pollution_with_control = []

    # Simulation without control
    agent_no_control = PollutionMonitorAgent(grid_size=10)
    for step in range(time_steps):
        for x, y in sources:
            intensity = np.random.uniform(10, 20)
            agent_no_control.add_pollution_source(x, y, intensity)
        agent_no_control.diffuse_pollution()
        avg_pollution_no_control.append(agent_no_control.get_average_pollution())

    # Simulation with control
    agent_with_control = PollutionMonitorAgent(grid_size=10)
    for step in range(time_steps):
        for x, y in sources:
            intensity = np.random.uniform(10, 20)
            agent_with_control.add_pollution_source(x, y, intensity)

        agent_with_control.diffuse_pollution()

        # Apply mitigation every 5 steps
        if step % 5 == 0:
            alerts = agent_with_control.detect_alerts()
            mitigation_zones = [(x, y) for x, y, _ in alerts]
            agent_with_control.apply_mitigation(mitigation_zones)

        avg_pollution_with_control.append(agent_with_control.get_average_pollution())

    return avg_pollution_no_control, avg_pollution_with_control, agent_with_control

def main():
    print("=" * 50)
    print("Pollution Control Monitor Agent")
    print("=" * 50)

    print("\nSimulating pollution over 50 time steps...")

    no_control, with_control, final_agent = simulate_pollution_control()

    avg_no_control = np.mean(no_control)
    avg_with_control = np.mean(with_control)
    reduction = ((avg_no_control - avg_with_control) / avg_no_control) * 100

    print(f"\nAverage pollution (no control): {avg_no_control:.2f}")
    print(f"Average pollution (with control): {avg_with_control:.2f}")
    print(f"Pollution reduction: {reduction:.2f}%")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Pollution over time
    ax1 = axes[0, 0]
    time_steps = range(len(no_control))
    ax1.plot(time_steps, no_control, label='No Control', color='red', linewidth=2)
    ax1.plot(time_steps, with_control, label='With Control', color='green', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Average Pollution Level')
    ax1.set_title('Pollution Levels Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final pollution grid
    ax2 = axes[0, 1]
    im = ax2.imshow(final_agent.pollution_grid, cmap='RdYlGn_r', vmin=0, vmax=100)
    ax2.set_title('Final Pollution Distribution (With Control)')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    plt.colorbar(im, ax=ax2, label='Pollution Level')

    # Plot 3: Average comparison
    ax3 = axes[1, 0]
    methods = ['No Control', 'With Control']
    averages = [avg_no_control, avg_with_control]
    colors = ['red', 'green']

    bars = ax3.bar(methods, averages, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Pollution Level')
    ax3.set_title('Control Method Comparison')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.1f}', ha='center', va='bottom')

    # Plot 4: Pollution reduction over time
    ax4 = axes[1, 1]
    reduction_over_time = [(nc - wc) for nc, wc in zip(no_control, with_control)]
    ax4.plot(time_steps, reduction_over_time, color='blue', linewidth=2)
    ax4.fill_between(time_steps, 0, reduction_over_time, alpha=0.3, color='blue')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Pollution Reduction')
    ax4.set_title('Pollution Reduction Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('pollution_control.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'pollution_control.png'")
    plt.show()

if __name__ == "__main__":
    main()

