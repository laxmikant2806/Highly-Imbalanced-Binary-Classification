"""
Advanced feature engineering for fraud detection.
Includes time features, amount normalization, and outlier handling.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours


def create_time_features(df):
    """
    Create cyclical time features from the Time column.
    Time is in seconds from the first transaction.
    """
    df = df.copy()
    
    # Convert seconds to hours (assuming Time represents seconds since first transaction)
    hours = (df['Time'] / 3600) % 24  # Hour of day (0-24)
    
    # Cyclical encoding using sin/cos
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    
    # Day segment (morning, afternoon, evening, night)
    day_fraction = hours / 24
    df['day_sin'] = np.sin(2 * np.pi * day_fraction)
    df['day_cos'] = np.cos(2 * np.pi * day_fraction)
    
    return df


def handle_amount_outliers(df, upper_percentile=99.9):
    """
    Handle outliers in Amount by capping at percentile thresholds.
    """
    df = df.copy()
    
    upper_limit = np.percentile(df['Amount'], upper_percentile)
    df['Amount'] = df['Amount'].clip(upper=upper_limit)
    
    # Create log-transformed amount (handles skewness)
    df['Amount_log'] = np.log1p(df['Amount'])
    
    return df


def create_amount_features(df):
    """
    Create additional amount-based features.
    """
    df = df.copy()
    
    # Standardized amount
    amount_mean = df['Amount'].mean()
    amount_std = df['Amount'].std()
    df['Amount_zscore'] = (df['Amount'] - amount_mean) / (amount_std + 1e-8)
    
    # Amount bins (categorical)
    df['Amount_bin'] = np.digitize(
        df['Amount'], 
        bins=[0, 10, 50, 100, 500, 1000, float('inf')]
    )
    
    return df


def preprocess_features(df, config):
    """
    Full feature preprocessing pipeline.
    """
    df = df.copy()
    
    # Handle Amount outliers
    df = handle_amount_outliers(df, upper_percentile=99.9)
    
    # Create time features
    df = create_time_features(df)
    
    # Create amount features
    df = create_amount_features(df)
    
    return df


def split_data(X, y, config):
    """Splits the data into training, validation, and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data_processing']['test_size'],
        stratify=y,
        random_state=config['data_processing']['random_state']
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config['data_processing']['val_size'],
        stratify=y_train,
        random_state=config['data_processing']['random_state']
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, method='standard'):
    """
    Scales the features using specified scaler.
    
    Args:
        method: 'standard' for StandardScaler, 'robust' for RobustScaler
    """
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def get_sampler(config):
    """
    Get the resampling strategy based on configuration.
    
    Supported strategies:
    - smote: Standard SMOTE
    - borderline_smote: Borderline-SMOTE (focuses on decision boundary)
    - adasyn: ADASYN (adaptive synthetic sampling)
    - smote_tomek: SMOTE + Tomek Links
    - smote_enn: SMOTE + Edited Nearest Neighbors
    """
    strategy = config['data_processing']['resampling'].get('strategy', 'smote_tomek')
    sampling_strategy = config['data_processing']['resampling']['smote']['sampling_strategy']
    random_state = config['data_processing']['random_state']
    
    if strategy == 'smote':
        return SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5
        )
    
    elif strategy == 'borderline_smote':
        return BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5,
            kind='borderline-1'
        )
    
    elif strategy == 'adasyn':
        return ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_neighbors=5
        )
    
    elif strategy == 'smote_enn':
        return SMOTEENN(
            smote=SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            ),
            enn=EditedNearestNeighbours(sampling_strategy='majority')
        )
    
    else:  # Default: smote_tomek
        return SMOTETomek(
            smote=SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            ),
            tomek=TomekLinks(sampling_strategy='majority')
        )


def resample_data(X_train_scaled, y_train, config):
    """Resamples the training data using the configured strategy."""
    sampler = get_sampler(config)
    
    print(f"Using resampling strategy: {type(sampler).__name__}")
    print(f"Before resampling: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
    
    print(f"After resampling: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    
    return X_resampled, y_resampled
