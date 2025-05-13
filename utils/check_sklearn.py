try:
    import sklearn
    print(f"sklearn is installed. Version: {sklearn.__version__}")
    print(f"sklearn path: {sklearn.__path__}")
except ImportError as e:
    print(f"Error importing sklearn: {e}")

try:
    from sklearn import neighbors
    print("sklearn.neighbors is available")
except ImportError as e:
    print(f"Error importing sklearn.neighbors: {e}")

import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
