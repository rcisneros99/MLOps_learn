"""
Data processing module for wine dataset
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    """
    Load the wine dataset and split it into training and test sets
    
    Args:
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_data, test_data, train_labels, test_labels)
    """
    X, y = datasets.load_wine(return_X_y=True)
    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return train_data, test_data, train_labels, test_labels 