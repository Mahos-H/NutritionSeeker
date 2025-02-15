import pandas as pd
import os
import numpy as np

def safe_convert(x):
    try:
        return float(x)
    except:
        if isinstance(x, str):
            if "/" in x:
                parts = x.split("/")
                try:
                    nums = [float(p.strip()) for p in parts if p.strip() != '']
                    return sum(nums) / len(nums) if nums else np.nan
                except:
                    return np.nan
            else:
                try:
                    x_clean = ''.join(c for c in x if c.isdigit() or c in ['.', '-'])
                    return float(x_clean) if x_clean != '' else np.nan
                except:
                    return np.nan
        return np.nan

def load_food_nutrition_mapping(nutrition_dir, target_columns=None):
    if target_columns is None:
        target_columns = ["Caloric Value", "Fat", "Carbohydrates"]
    
    dataframes = []
    for filename in os.listdir(nutrition_dir):
        if filename.endswith(".csv") and filename.startswith("FOOD-DATA-GROUP"):
            file_path = os.path.join(nutrition_dir, filename)
            df = pd.read_csv(file_path)
            df['Source'] = 'food_nutrition'
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    if "food" in combined_df.columns:
        combined_df["FoodName"] = combined_df["food"].astype(str).str.lower().str.strip()
    elif "item" in combined_df.columns:
        combined_df["FoodName"] = combined_df["item"].astype(str).str.lower().str.strip()
    else:
        raise ValueError("Neither 'food' nor 'item' column found in food nutrition data.")
    
    for col in target_columns:
        if col not in combined_df.columns:
            raise ValueError(f"Column '{col}' not found in food nutrition data.")
    
    grouped = combined_df.groupby("FoodName")[target_columns].mean().reset_index()
    mapping = {}
    for _, row in grouped.iterrows():
        food_name = row["FoodName"]
        target_vector = row[target_columns].values.astype(np.float32)
        mapping[food_name] = target_vector
    return mapping, target_columns
