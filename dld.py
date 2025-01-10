import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    def __init__(self):
        # Predefined data patterns and distributions for different domains
        self.patterns = {
            'age': {'min': 0, 'max': 100, 'distribution': 'normal', 'mean': 35, 'std': 15},
            'salary': {'min': 20000, 'max': 200000, 'distribution': 'lognormal', 'mean': 11, 'std': 0.5},
            'temperature': {'min': -20, 'max': 45, 'distribution': 'normal', 'mean': 20, 'std': 10},
            'humidity': {'min': 0, 'max': 100, 'distribution': 'normal', 'mean': 60, 'std': 15},
            'price': {'min': 0, 'max': 1000, 'distribution': 'lognormal', 'mean': 4, 'std': 1},
            'rating': {'min': 1, 'max': 5, 'distribution': 'normal', 'mean': 3.5, 'std': 0.8},
            'probability': {'min': 0, 'max': 1, 'distribution': 'beta', 'a': 2, 'b': 2}
        }
        
        # Category mappings for categorical data
        self.categories = {
            'gender': ['Male', 'Female', 'Other'],
            'education': ['High School', 'Bachelor', 'Master', 'PhD'],
            'department': ['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'],
            'country': ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan'],
            'status': ['Active', 'Inactive', 'Pending'],
            'category': ['A', 'B', 'C', 'D'],
            'color': ['Red', 'Blue', 'Green', 'Yellow', 'Black', 'White']
        }

    def generate_numeric_data(self, n_samples, feature_spec):
        """
        Generate numeric data based on distribution and range specifications
        
        Parameters:
        n_samples: int - number of samples to generate
        feature_spec: dict - specification for the feature including distribution and parameters
        """
        if feature_spec['distribution'] == 'normal':
            data = np.random.normal(feature_spec['mean'], feature_spec['std'], n_samples)
            
        elif feature_spec['distribution'] == 'lognormal':
            data = np.random.lognormal(feature_spec['mean'], feature_spec['std'], n_samples)
            
        elif feature_spec['distribution'] == 'uniform':
            data = np.random.uniform(feature_spec['min'], feature_spec['max'], n_samples)
            
        elif feature_spec['distribution'] == 'beta':
            data = np.random.beta(feature_spec['a'], feature_spec['b'], n_samples)
            
        # Clip values to specified range
        data = np.clip(data, feature_spec['min'], feature_spec['max'])
        return data

    def generate_categorical_data(self, n_samples, categories, weights=None):
        """Generate categorical data from given categories"""
        return np.random.choice(categories, size=n_samples, p=weights)

    def generate_datetime_data(self, n_samples, start_date=None, end_date=None):
        """Generate datetime data within a specified range"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
            
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        
        timestamps = np.random.uniform(start_ts, end_ts, n_samples)
        return [datetime.fromtimestamp(ts) for ts in timestamps]

    def generate_dataset(self, n_samples, features, custom_ranges=None):
        """
        Generate a complete dataset based on feature specifications
        
        Parameters:
        n_samples: int - number of samples to generate
        features: list - list of feature names
        custom_ranges: dict - custom specifications for features (optional)
        """
        data = {}
        custom_ranges = custom_ranges or {}
        
        for feature in features:
            # Check if custom range is provided
            if feature in custom_ranges:
                spec = custom_ranges[feature]
                
                if 'categories' in spec:
                    data[feature] = self.generate_categorical_data(
                        n_samples, 
                        spec['categories'],
                        spec.get('weights', None)
                    )
                elif 'datetime' in spec:
                    data[feature] = self.generate_datetime_data(
                        n_samples,
                        spec.get('start_date'),
                        spec.get('end_date')
                    )
                else:
                    data[feature] = self.generate_numeric_data(n_samples, spec)
                    
            # Use predefined patterns if available
            elif feature.lower() in self.patterns:
                data[feature] = self.generate_numeric_data(
                    n_samples,
                    self.patterns[feature.lower()]
                )
                
            # Use predefined categories if available
            elif feature.lower() in self.categories:
                data[feature] = self.generate_categorical_data(
                    n_samples,
                    self.categories[feature.lower()]
                )
                
            # Default to uniform distribution
            else:
                default_spec = {
                    'min': 0,
                    'max': 100,
                    'distribution': 'uniform'
                }
                data[feature] = self.generate_numeric_data(n_samples, default_spec)
                
        return pd.DataFrame(data)
    

# Create an instance of the generator
generator = SyntheticDataGenerator()

# Generate a sample dataset
features = ['age', 'salary', 'department', 'gender']
df = generator.generate_dataset(n_samples=5, features=features)

# Display the generated data
print(df)

print(df)  # Print all data
print(df.head())  # Print first 5 rows
df.to_csv('generated_data.csv')  # Save as CSV
print(df.describe())  # Statistical summary

# Example 2: Generate dataset with custom specifications
custom_ranges = {
    'temperature': {
        'min': 15,
        'max': 35,
        'distribution': 'normal',
        'mean': 25,
        'std': 5
    },
    'city': {
        'categories': ['New York', 'London', 'Tokyo', 'Paris'],
        'weights': [0.3, 0.3, 0.2, 0.2]
    },
    'timestamp': {
        'datetime': True,
        'start_date': datetime(2023, 1, 1),
        'end_date': datetime(2024, 1, 1)
    }
}

df = generator.generate_dataset(
    n_samples=1000,
    features=['temperature', 'city', 'timestamp'],
    custom_ranges=custom_ranges
)
