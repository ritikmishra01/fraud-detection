import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_cleaned_data():
    df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')

    # 🔥 FORCE BOTH CLASSES
    fraud = df[df['Class'] == 1]      # all fraud (~492 rows)
    normal = df[df['Class'] == 0].sample(2000, random_state=42)

    df = pd.concat([fraud, normal])

    # Shuffle data
    df = df.sample(frac=1, random_state=42)

    # Scale Amount
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Drop Time
    df = df.drop(['Time'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    return train_test_split(X, y, test_size=0.2, random_state=42)
