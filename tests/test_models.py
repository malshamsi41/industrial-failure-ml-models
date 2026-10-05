import pandas as pd
from src.train_models import features
def test_features_are_leakage_free():
    df=pd.read_csv("data/raw/ai4i2020.csv"); X=features(df)
    assert set(["TWF","HDF","PWF","OSF","RNF","Machine failure","UDI","Product ID"]).isdisjoint(X.columns)
