import pandas as pd
import numpy as np
import pickle


def save_obj(obj, name):
    with open('resources/objects/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('resources/objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def get_categories(df):
    categories_dict = {}
    categories_raw = df.dtypes
    example_data = df.sample()
    for category in categories_raw.index:
        numeric_data_types = {int, float, complex}
        cat_type = categories_raw[category]
        if cat_type not in numeric_data_types and not pd.api.types.is_numeric_dtype(cat_type):
            example_value = df.sample()[category].values[0]
            categories_dict[category] = (str, example_value)
        else:
            example_value = df.sample()[category].values[0]
            categories_dict[category] = (float, example_value)

    return categories_dict