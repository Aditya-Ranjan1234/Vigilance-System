try:
    import vigilance_system
    print("vigilance_system module imported successfully")
except ImportError as e:
    print(f"Error importing vigilance_system: {e}")

try:
    from vigilance_system.detection.ml_algorithms import KNNTracker, SVMTracker
    print("KNNTracker and SVMTracker imported successfully")
except ImportError as e:
    print(f"Error importing KNNTracker and SVMTracker: {e}")

try:
    from vigilance_system.alert.decision_maker import decision_maker
    print("decision_maker imported successfully")
except ImportError as e:
    print(f"Error importing decision_maker: {e}")

import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
