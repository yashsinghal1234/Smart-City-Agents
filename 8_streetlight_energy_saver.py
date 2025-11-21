"""
Streetlight Energy Saver Agent
Adaptive brightness control based on motion detection
"""
import matplotlib.pyplot as plt
import numpy as np

class StreetlightAgent:
    def __init__(self, light_id, location):
        self.id = light_id
        self.location = location
        self.brightness = 30  # Default dim (30%)
        self.max_brightness = 100
        self.min_brightness = 20

    def detect_motion(self, pedestrians):
        """Check if motion detected nearby"""
        detection_radius = 2.0

        for ped in pedestrians:
            distance = np.sqrt((self.location[0] - ped[0])**2 +
                             (self.location[1] - ped[1])**2)
            if distance < detection_radius:
                return True
        return False

    def adjust_brightness(self, motion_detected):
        """Adjust brightness based on motion"""
        if motion_detected:
            self.brightness = self.max_brightness
        else:
            self.brightness = self.min_brightness
        return self.brightness

    def get_energy_consumption(self):
        """Calculate energy consumption (watts)"""
        return self.brightness * 0.5  # 50W at 100% brightness

def simulate_streetlights(time_steps=100, adaptive=False):
    """Simulate streetlight system"""
    np.random.seed(42)

    # Create streetlights in a grid
    num_lights = 20
    lights = []
    for i in range(num_lights):
        x = (i % 5) * 3
        y = (i // 5) * 3
        lights.append(StreetlightAgent(i, (x, y)))

    total_energy = 0
    energy_per_step = []

    for step in range(time_steps):
        # Generate random pedestrians
        num_pedestrians = np.random.randint(0, 8)
        pedestrians = [(np.random.rand() * 15, np.random.rand() * 12)
                      for _ in range(num_pedestrians)]

        step_energy = 0

        for light in lights:
            if adaptive:
                motion = light.detect_motion(pedestrians)
                light.adjust_brightness(motion)
            else:
                light.brightness = 100  # Always full brightness

            step_energy += light.get_energy_consumption()

        total_energy += step_energy
        energy_per_step.append(step_energy)

    return total_energy, energy_per_step, lights

def main():
    print("=" * 50)
    print("Streetlight Energy Saver Agent")
    print("=" * 50)

    time_steps = 100

    # Simulate without adaptive control
    print("\nSimulating traditional streetlights (always on)...")
    energy_traditional, steps_traditional, _ = simulate_streetlights(
        time_steps=time_steps, adaptive=False
    )

    # Simulate with adaptive control
    print("Simulating adaptive streetlights (motion-based)...")
    energy_adaptive, steps_adaptive, lights = simulate_streetlights(
        time_steps=time_steps, adaptive=True
    )

    energy_saved = energy_traditional - energy_adaptive
    savings_percent = (energy_saved / energy_traditional) * 100

    print(f"\nTotal energy (traditional): {energy_traditional:.2f} kWh")
    print(f"Total energy (adaptive): {energy_adaptive:.2f} kWh")
    print(f"Energy saved: {energy_saved:.2f} kWh ({savings_percent:.2f}%)")

    # Cost savings (assume $0.12 per kWh)
    cost_per_kwh = 0.12
    cost_saved = energy_saved * cost_per_kwh / 1000
    print(f"Cost saved: ${cost_saved:.2f}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Energy consumption over time
    ax1 = axes[0, 0]
    ax1.plot(steps_traditional, label='Traditional', color='orange', linewidth=2)
    ax1.plot(steps_adaptive, label='Adaptive', color='green', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Energy Consumption (W)')
    ax1.set_title('Energy Consumption Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total energy comparison
    ax2 = axes[0, 1]
    methods = ['Traditional\n(Always On)', 'Adaptive\n(Motion-Based)']
    energies = [energy_traditional, energy_adaptive]
    colors = ['orange', 'green']

    bars = ax2.bar(methods, energies, color=colors, alpha=0.7)
    ax2.set_ylabel('Total Energy (kWh)')
    ax2.set_title('Total Energy Consumption Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.0f}', ha='center', va='bottom')

    # Plot 3: Streetlight grid
    ax3 = axes[1, 0]
    for light in lights:
        color_intensity = light.brightness / 100
        ax3.scatter(light.location[0], light.location[1],
                   c=[(1, 1-color_intensity, 0)], s=300, marker='o',
                   edgecolors='black', linewidth=2)
        ax3.text(light.location[0], light.location[1], str(light.id),
                ha='center', va='center', fontsize=8)

    ax3.set_title('Streetlight Grid (Final State)')
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Savings metrics
    ax4 = axes[1, 1]
    metrics = ['Energy\nSaved (kWh)', f'Cost\nSaved ($)', 'Savings\n(%)']
    values = [energy_saved, cost_saved, savings_percent]
    colors_metrics = ['green', 'blue', 'purple']

    bars = ax4.bar(metrics, values, color=colors_metrics, alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Energy Savings Metrics')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('graphs/streetlight_energy_saver.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'graphs/streetlight_energy_saver.png'")
    plt.show()

if __name__ == "__main__":
    main()
