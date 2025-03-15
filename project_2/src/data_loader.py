import numpy as np
import os
from typing import Optional, Tuple, Dict
from PIL import Image


class DataLoader:
    def __init__(self, data_dir: Optional[str] = None) -> None:
        """
        Initialize the DataLoader with a specified data directory.
        """
        if data_dir is None:
            self.data_dir = r'C:\Users\askha\Documents\Math Mod\MathModelling-25\project_2\data'
        else:
            self.data_dir = data_dir
        self.feature_scalers = None
    
    def load_to_np(self, input_data: str) -> np.ndarray:
        """
        Load data from a file into a numpy array.
        """
        current_dir = os.getcwd()
        try:
            os.chdir(self.data_dir)
            data = np.genfromtxt(input_data, delimiter=',')
        finally:
            os.chdir(current_dir)
        return data
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features to [0, 1] range using min-max normalization.
        """
        if fit:
            # Calculate min and max for each feature
            self.feature_scalers = {
                'min': np.min(X, axis=0),
                'max': np.max(X, axis=0)
            }
        
        if self.feature_scalers is None:
            raise ValueError("No feature scalers available. Call with fit=True first.")
        
        # Avoid division by zero for constant features
        denominator = self.feature_scalers['max'] - self.feature_scalers['min']
        denominator[denominator == 0] = 1.0
        
        # Apply min-max scaling
        X_scaled = (X - self.feature_scalers['min']) / denominator
        
        return X_scaled
    
    def get_train_test_val_split(self, 
                                 input_data: str,
                                 feature_cols: Optional[np.ndarray] = None, 
                                 target_cols: Optional[np.ndarray] = None,
                                 train_ratio: float = 0.8, 
                                 test_ratio: float = 0.1,
                                 val_ratio: float = 0.1,
                                 shuffle: bool = True,
                                 random_seed: Optional[int] = 69,
                                 scale_features: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load and split data into training, testing, and validation sets.
        """
        # Load the data
        data = self.load_to_np(input_data)
        
        # Set default feature and target columns if not specified
        if feature_cols is None and target_cols is None:
            # Default: all columns except the last one are features
            feature_cols = np.arange(data.shape[1] - 1)
            target_cols = np.array([data.shape[1] - 1])
            
        # Extract features and targets
        X = data[:, feature_cols]
        y = data[:, target_cols]
        
        # Validate ratios
        assert np.isclose(train_ratio + test_ratio + val_ratio, 1.0), "Ratios must sum to 1"
        
        # Shuffle data
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        # Calculate split indices
        n_samples = len(X)
        train_end = int(train_ratio * n_samples)
        test_end = train_end + int(test_ratio * n_samples)
        
        # Split data
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        X_val, y_val = X[test_end:], y[test_end:]
        
        # Scale features if requested
        if scale_features:
            X_train = self.scale_features(X_train, fit=True)
            X_test = self.scale_features(X_test, fit=False)
            X_val = self.scale_features(X_val, fit=False)
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'val': (X_val, y_val)
        }
    
    
def load_image_to_array(image_path: str) -> np.ndarray:
            """
            Load a black and white PNG image and return an array of the coordinates of each pixel and its value (1 for white, 0 for black).
            """
            # Open the image
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Convert image to numpy array
            image_array = np.array(image)
            
            # Get the coordinates of each pixel
            coords = np.column_stack(np.where(image_array < 128))  # Assuming threshold of 128 for black and white
            
            # Get the values of each pixel (1 for white, 0 for black)
            values = (image_array[coords[:, 0], coords[:, 1]] > 128).astype(int)
            
            # Combine coordinates and values
            result = np.column_stack((coords, values))
            
            return result
