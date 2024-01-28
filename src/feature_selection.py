# feature_selection.py

from sklearn.preprocessing import LabelEncoder

def label_encode(df, column):
    # Function for Label Encoding
    le = LabelEncoder()
    df[column+'_n'] = le.fit_transform(df[column])
    return df
