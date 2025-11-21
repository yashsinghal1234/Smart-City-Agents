"""
Emergency Response Dispatcher Agent
Uses nearest-neighbor algorithm for optimal emergency response
"""
import matplotlib.pyplot as plt
import numpy as np

class EmergencyDispatcher:
    def __init__(self):
        self.ambulances = []
        self.incidents = []

    def add_ambulance(self, ambulance_id, location):
        """Add an ambulance with location"""
        self.ambulances.append({
            'id': ambulance_id,
            'location': location,
            'available': True
        })

    def add_incident(self, incident_id, location, severity):
        """Add an incident"""
        self.incidents.append({
            'id': incident_id,
            'location': location,
            'severity': severity,
            'assigned': None,
            'response_time': None
        })

    def calculate_distance(self, loc1, loc2):
        """Calculate Euclidean distance"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    def dispatch_nearest(self):
        """Dispatch ambulances to nearest incidents"""
        # Sort incidents by severity (higher first)
        sorted_incidents = sorted(self.incidents, key=lambda x: x['severity'], reverse=True)

        for incident in sorted_incidents:
            min_distance = float('inf')
            best_ambulance = None

            for ambulance in self.ambulances:
                if ambulance['available']:
                    distance = self.calculate_distance(
                        ambulance['location'], incident['location']
                    )
                    if distance < min_distance:
                        min_distance = distance
                        best_ambulance = ambulance

            if best_ambulance:
                incident['assigned'] = best_ambulance['id']
                incident['response_time'] = min_distance / 0.6  # 0.6 units/min = ~60 km/h
                best_ambulance['available'] = False

        return self.incidents

    def dispatch_random(self):
        """Random dispatch for comparison"""
        available = [a for a in self.ambulances if a['available']]

        for incident in self.incidents:
            if available:
                ambulance = available.pop(0)
                distance = self.calculate_distance(
                    ambulance['location'], incident['location']
                )
                incident['assigned'] = ambulance['id']
                incident['response_time'] = distance / 0.6

        return self.incidents

def main():
    print("=" * 50)
    print("Emergency Response Dispatcher Agent")
    print("=" * 50)

    np.random.seed(42)

    # Create dispatcher
    dispatcher = EmergencyDispatcher()

    # Add ambulances
    num_ambulances = 8
    for i in range(num_ambulances):
        location = (np.random.rand() * 10, np.random.rand() * 10)
        dispatcher.add_ambulance(i, location)

    # Add incidents
    num_incidents = 6
    for i in range(num_incidents):
        location = (np.random.rand() * 10, np.random.rand() * 10)
        severity = np.random.randint(1, 4)  # 1=low, 3=high
        dispatcher.add_incident(i, location, severity)

    print(f"\nAmbulances available: {num_ambulances}")
    print(f"Incidents reported: {num_incidents}")

    # Optimal dispatch
    dispatcher.dispatch_nearest()

    assigned_incidents = [i for i in dispatcher.incidents if i['assigned'] is not None]
    avg_response = np.mean([i['response_time'] for i in assigned_incidents])

    print(f"\nIncidents assigned: {len(assigned_incidents)}")
    print(f"Average response time: {avg_response:.2f} minutes")

    print("\nDispatch Details:")
    for incident in dispatcher.incidents:
        if incident['assigned'] is not None:
            print(f"Incident {incident['id']} (Severity: {incident['severity']}) "
                  f"-> Ambulance {incident['assigned']} "
                  f"(ETA: {incident['response_time']:.1f} min)")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Dispatch map
    ax1 = axes[0]

    # Plot ambulances
    for ambulance in dispatcher.ambulances:
        color = 'green' if ambulance['available'] else 'gray'
        ax1.scatter(ambulance['location'][0], ambulance['location'][1],
                   c=color, s=200, marker='^', edgecolors='black',
                   label='Ambulance' if ambulance['id'] == 0 else '')

    # Plot incidents
    severity_colors = {1: 'yellow', 2: 'orange', 3: 'red'}
    for incident in dispatcher.incidents:
        color = severity_colors[incident['severity']]
        ax1.scatter(incident['location'][0], incident['location'][1],
                   c=color, s=150, marker='*', edgecolors='black')

        # Draw dispatch line
        if incident['assigned'] is not None:
            ambulance = next(a for a in dispatcher.ambulances if a['id'] == incident['assigned'])
            ax1.plot([ambulance['location'][0], incident['location'][0]],
                    [ambulance['location'][1], incident['location'][1]],
                    'b--', alpha=0.5, linewidth=2)

    ax1.set_title('Emergency Response Dispatch Map')
    ax1.set_xlabel('X coordinate (km)')
    ax1.set_ylabel('Y coordinate (km)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Ambulance', 'Low Severity', 'Medium Severity', 'High Severity'])

    # Plot 2: Response times
    ax2 = axes[1]

    incident_ids = [f"I{i['id']}" for i in assigned_incidents]
    response_times = [i['response_time'] for i in assigned_incidents]
    severities = [i['severity'] for i in assigned_incidents]
    colors = [severity_colors[s] for s in severities]

    bars = ax2.bar(incident_ids, response_times, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=avg_response, color='blue', linestyle='--',
                label=f'Average: {avg_response:.1f} min')
    ax2.set_xlabel('Incident')
    ax2.set_ylabel('Response Time (minutes)')
    ax2.set_title('Emergency Response Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('emergency_response.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'emergency_response.png'")
    plt.show()

if __name__ == "__main__":
    main()

