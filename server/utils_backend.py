import json
import pickle
import numpy as np


__area_types = None
__locations = None
__data_columns = None
__model = None
def get_estimated_prices(location, total_sqft, bath, bedroom, area_type):
    global __data_columns
    
    if not area_type:
      raise ValueError("area_type is not selected.")

    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bedroom

    loc_col = f"location_{location.strip().lower().replace(' ','_')}"
    area_col = f"area_type_{area_type.strip().lower().replace(' ','_')}"

    if loc_col in __data_columns:
        x[__data_columns.index(loc_col)] = 1
    if area_col in __data_columns:
        x[__data_columns.index(area_col)] = 1
    

    return round(__model.predict([x])[0], 2)

   
def get_location_names():
    return [loc.replace("location_", "").title() for loc in __locations]

def get_area_types():
    return [area.replace("area_type_", "").title() for area in __area_types]



def load_saved_artifacts():
    print("Loading saved artifacts...")

    global __data_columns
    global __locations
    global __area_types
    global __model

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __area_types = [col.replace("area_type_", "").replace("_", " ").title()
                        for col in __data_columns if col.startswith("area_type_")]
        __locations = [col.replace("location_", "").replace("_", " ").title()
                       for col in __data_columns if col.startswith("location_")]

    with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)

    print("Loading saved artifacts...done")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_area_types())

