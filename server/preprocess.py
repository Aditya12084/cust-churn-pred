# server/preprocess.py
import pandas as pd
import json

#loading expected columns
with open('../model/columns.json', 'r') as f:
    EXPECTED_COLUMNS = json.load(f)

def transform_input(data_dict):
    df = pd.DataFrame([data_dict])

    df_encoded = pd.get_dummies(df, drop_first=True)

    df_aligned = df_encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    return df_aligned.astype(float)