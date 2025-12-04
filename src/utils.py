
import pandas as pd
def prepare_input(df, template_columns):
    df_enc = pd.get_dummies(df)
    df_enc = df_enc.reindex(columns=template_columns, fill_value=0)
    return df_enc
