import pandas as pd
import numpy as np

def clean_diabetic_data(file_path):
    # 1. Load the dataset
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    
    # 2. Replace '?' placeholders with actual NaN values
    df.replace(to_replace=r'.*\?.*', value=np.nan, regex=True, inplace=True)

    # 3. Impute Missing Values (The "Average" logic)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns (like time_in_hospital), use MEAN
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                print(f"Filled numeric {col} with mean: {mean_val:.2f}")
            else:
                # For categorical columns (like race, weight), use MODE (most frequent)
                # mode() returns a series, so we take the first element [0]
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    print(f"Filled categorical {col} with mode: {mode_val}")

    # 4. Remove invalid gender entries (Standard cleanup)
    df = df[df['gender'] != 'Unknown/Invalid']
    
    # 5. Convert 'age' to midpoints (Numeric)
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
        '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)
    
    # 6. Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

# Execute
cleaned_df = clean_diabetic_data('./data/diabetic_data.csv')

# Verify results
print(f"Final shape: {cleaned_df.shape}")
print(f"Remaining nulls: {cleaned_df.isna().sum().sum()}")

# Save

cleaned_df.to_csv('./data/cleaned_data.csv', index=False)