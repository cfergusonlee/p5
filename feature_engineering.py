import numpy as np

# Creates the following additional features:
# - fract_to_poi
# - fract_from_poi
# - salary_over_bonus
# - exer_stock_opts_over_tot
def create_features(data_dict):
    for key, value in data_dict.items():
        from_person_to_poi = value['from_this_person_to_poi']
        from_messages = value['from_messages']
        from_poi_to_this_person = value['from_poi_to_this_person']
        to_messages = value['to_messages']
        salary = value['salary']
        bonus = value['bonus']
        exercised_stock_options = value['exercised_stock_options']
        total_stock_value = value['total_stock_value']
        
        if from_person_to_poi != 'NaN' and from_messages != 'NaN':
            value['fract_to_poi'] = float(from_person_to_poi)/from_messages
        else:
            value['fract_to_poi'] = 'NaN'
        if from_poi_to_this_person != 'NaN' and to_messages != 'NaN':
            value['fract_from_poi'] = float(from_poi_to_this_person)/to_messages
        else:
            value['fract_from_poi'] = 'NaN'
        if salary != 'NaN' and bonus != 'NaN':
            value['salary_over_bonus'] = float(salary)/bonus
        else:
            value['salary_over_bonus'] = 'NaN'
        if exercised_stock_options != 'NaN' and total_stock_value != 'NaN' and total_stock_value != 0:
            value['exer_stock_opts_over_tot'] = float(exercised_stock_options)/total_stock_value
        else:
            value['exer_stock_opts_over_tot'] = 'NaN'
    return data_dict

# Creates a MinMaxScaler to scale all features from 0 to 1
def scale_features(data_dict):
    import pandas as pd
    
    
    # Create dataframe from dictionary
    data_df = pd.DataFrame.from_dict(data_dict, orient='index')
    data_df.replace('NaN', np.nan, inplace = True)
    
    # Scale using manual scaler to ignore non-numerical values
    scaled_df = data_df.apply(manual_MinMaxScaler)
    
    # Return NaN values to original form and create .csv to check
    scaled_df.replace(np.nan, 'NaN', inplace = True)
    scaled_df.to_csv('scaled_enron_data.csv')
    return pd.DataFrame.to_dict(scaled_df, orient='index')
    
def manual_MinMaxScaler(df):
    if df.name != 'name' and df.name != 'email_address' and df.name != 'poi':
        min_val = df.min()
        max_val = df.max()
        return (df-min_val)/(max_val-min_val)
    else:
        return df

