import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_cleaned_data(file_path):
    df = pd.read_csv(file_path)

    # Take small sample for fast startup (IMPORTANT)
    df = df.sample(5000)

    # Scale Amount
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Drop Time column
    df = df.drop(['Time'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    return train_test_split(X, y, test_size=0.2, random_state=42)