import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_cleaned_data():
    try:
        # Try loading online dataset
        df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
    except:
        # Fallback small dataset (safe)
        data = {
            'V1': [0,1,2,3,4],
            'V2': [1,2,3,4,5],
            'V3': [2,3,4,5,6],
            'V4': [3,4,5,6,7],
            'V5': [1,1,1,1,1],
            'V6': [0,0,0,0,0],
            'V7': [1,2,1,2,1],
            'V8': [0,1,0,1,0],
            'V9': [1,0,1,0,1],
            'V10':[0,1,0,1,0],
            'V11':[1,1,1,1,1],
            'V12':[0,0,0,0,0],
            'V13':[1,2,3,4,5],
            'V14':[2,3,4,5,6],
            'V15':[1,1,1,1,1],
            'V16':[0,0,0,0,0],
            'V17':[1,2,3,4,5],
            'V18':[2,3,4,5,6],
            'V19':[1,1,1,1,1],
            'V20':[0,0,0,0,0],
            'V21':[1,2,3,4,5],
            'V22':[2,3,4,5,6],
            'V23':[1,1,1,1,1],
            'V24':[0,0,0,0,0],
            'V25':[1,2,3,4,5],
            'V26':[2,3,4,5,6],
            'V27':[1,1,1,1,1],
            'V28':[0,0,0,0,0],
            'Amount':[100,200,150,300,250],
            'Class':[0,0,1,0,1]
        }
        df = pd.DataFrame(data)

    # Reduce size
    df = df.sample(min(len(df), 3000), random_state=42)

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    if 'Time' in df.columns:
        df = df.drop(['Time'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    return train_test_split(X, y, test_size=0.2, random_state=42)
