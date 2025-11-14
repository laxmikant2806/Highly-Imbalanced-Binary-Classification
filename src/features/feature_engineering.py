from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

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

def scale_features(X_train, X_val, X_test):
    """Scales the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def resample_data(X_train_scaled, y_train, config):
    """Resamples the training data using SMOTE + Tomek links."""
    smote_tomek = SMOTETomek(
        smote=SMOTE(
            sampling_strategy=config['data_processing']['resampling']['smote']['sampling_strategy'],
            random_state=config['data_processing']['random_state']
        ),
        tomek=TomekLinks(sampling_strategy='majority')
    )
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)

    return X_resampled, y_resampled
