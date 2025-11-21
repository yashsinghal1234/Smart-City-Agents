"""
Waste Segregation AI Agent
Classifies waste items using simple ML classifier
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class WasteSegregationAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.categories = ['Plastic', 'Paper', 'Glass', 'Metal', 'Organic']

    def generate_training_data(self, num_samples=500):
        """Generate synthetic waste data"""
        # Features: weight, size, density, magnetic, decomposable
        X = []
        y = []

        for _ in range(num_samples):
            category = np.random.choice(len(self.categories))

            if category == 0:  # Plastic
                features = [
                    np.random.uniform(10, 100),    # light
                    np.random.uniform(5, 50),      # medium size
                    np.random.uniform(0.9, 1.5),   # low density
                    np.random.uniform(0, 0.1),     # not magnetic
                    np.random.uniform(0, 0.2)      # not decomposable
                ]
            elif category == 1:  # Paper
                features = [
                    np.random.uniform(5, 50),      # very light
                    np.random.uniform(10, 40),     # flat/medium
                    np.random.uniform(0.7, 1.2),   # very low density
                    np.random.uniform(0, 0.1),     # not magnetic
                    np.random.uniform(0.7, 1.0)    # highly decomposable
                ]
            elif category == 2:  # Glass
                features = [
                    np.random.uniform(50, 200),    # heavy
                    np.random.uniform(10, 60),     # various sizes
                    np.random.uniform(2.4, 2.8),   # high density
                    np.random.uniform(0, 0.1),     # not magnetic
                    np.random.uniform(0, 0.1)      # not decomposable
                ]
            elif category == 3:  # Metal
                features = [
                    np.random.uniform(30, 150),    # medium-heavy
                    np.random.uniform(5, 40),      # compact
                    np.random.uniform(2.5, 8.0),   # very high density
                    np.random.uniform(0.6, 1.0),   # magnetic
                    np.random.uniform(0, 0.1)      # not decomposable
                ]
            else:  # Organic
                features = [
                    np.random.uniform(20, 80),     # medium weight
                    np.random.uniform(10, 50),     # various sizes
                    np.random.uniform(0.8, 1.3),   # low density
                    np.random.uniform(0, 0.1),     # not magnetic
                    np.random.uniform(0.8, 1.0)    # highly decomposable
                ]

            X.append(features)
            y.append(category)

        return np.array(X), np.array(y)

    def train(self, X_train, y_train):
        """Train the classifier"""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict waste category"""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)

        return accuracy, conf_matrix, predictions

def main():
    print("=" * 50)
    print("Waste Segregation AI Agent")
    print("=" * 50)

    # Create agent
    agent = WasteSegregationAgent()

    # Generate training data
    print("\nGenerating synthetic waste data...")
    X, y = agent.generate_training_data(num_samples=1000)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    print("\nTraining waste segregation model...")
    agent.train(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    accuracy, conf_matrix, predictions = agent.evaluate(X_test, y_test)

    print(f"\nClassification Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions,
                                target_names=agent.categories))

    # Calculate per-category accuracy
    category_accuracies = []
    for i, category in enumerate(agent.categories):
        mask = y_test == i
        if mask.sum() > 0:
            cat_acc = (predictions[mask] == i).sum() / mask.sum()
            category_accuracies.append(cat_acc * 100)
            print(f"{category}: {cat_acc * 100:.2f}%")
        else:
            category_accuracies.append(0)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Confusion Matrix
    ax1 = axes[0, 0]
    im = ax1.imshow(conf_matrix, cmap='Blues')
    ax1.set_xticks(np.arange(len(agent.categories)))
    ax1.set_yticks(np.arange(len(agent.categories)))
    ax1.set_xticklabels(agent.categories, rotation=45, ha='right')
    ax1.set_yticklabels(agent.categories)

    # Add text annotations
    for i in range(len(agent.categories)):
        for j in range(len(agent.categories)):
            text = ax1.text(j, i, conf_matrix[i, j],
                          ha="center", va="center", color="black")

    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    plt.colorbar(im, ax=ax1)

    # Plot 2: Per-category accuracy
    ax2 = axes[0, 1]
    colors = ['blue', 'green', 'cyan', 'orange', 'brown']
    bars = ax2.bar(agent.categories, category_accuracies, color=colors, alpha=0.7)
    ax2.axhline(y=80, color='red', linestyle='--', label='Target (80%)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Per-Category Classification Accuracy')
    ax2.set_ylim([0, 105])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, category_accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom')

    # Plot 3: Feature importance
    ax3 = axes[1, 0]
    feature_names = ['Weight', 'Size', 'Density', 'Magnetic', 'Decomposable']
    importances = agent.model.feature_importances_

    bars = ax3.barh(feature_names, importances, color='purple', alpha=0.7)
    ax3.set_xlabel('Importance')
    ax3.set_title('Feature Importance for Classification')
    ax3.grid(True, alpha=0.3, axis='x')

    # Plot 4: Sample distribution
    ax4 = axes[1, 1]
    unique, counts = np.unique(y_test, return_counts=True)
    category_names = [agent.categories[i] for i in unique]

    ax4.pie(counts, labels=category_names, autopct='%1.1f%%',
           colors=colors[:len(unique)], startangle=90)
    ax4.set_title('Test Data Distribution by Category')

    plt.tight_layout()
    plt.savefig('waste_segregation.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved as 'waste_segregation.png'")
    plt.show()

if __name__ == "__main__":
    main()
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

