import pandas as pd
import numpy as np
import os

def preprocess_data(file_path):
    print(f"Preprocessing data from {file_path}...")
    df = pd.read_csv(file_path)

    # 1. Drop unnecessary columns
    # 'society' has many missing values, 'availability' might not be a strong predictor for price in this context
    # Keeping location for now but it needs encoding later.
    df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

    # 2. Handle missing values
    df3 = df2.dropna()

    # 3. Clean 'size' column (extract BHK)
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

    # 4. Clean 'total_sqft' (handle ranges)
    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    df4 = df3.copy()
    df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
    df4 = df4[df4.total_sqft.notnull()]

    # 5. Feature Engineering: Price per sqft
    df5 = df4.copy()
    df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

    # 6. Dimensionality Reduction for 'location'
    df5.location = df5.location.apply(lambda x: x.strip())
    location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

    # 7. Outlier Removal
    # Remove business logic outliers (e.g., less than 300 sqft per BHK)
    df6 = df5[~(df5.total_sqft / df5.bhk < 300)]

    # Remove price outliers using mean and standard deviation per location
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df7 = remove_pps_outliers(df6)

    # Remove BHK outliers (where a 2BHK is more expensive than a 3BHK in same area)
    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    df8 = remove_bhk_outliers(df7)

    # Final cleanup
    df9 = df8.drop(['size', 'price_per_sqft'], axis='columns')
    
    # Save the processed data
    output_path = os.path.join("data", "Processed_House_Data.csv")
    df9.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    return df9

if __name__ == "__main__":
    file_path = os.path.join("data", "House_Data.csv")
    if os.path.exists(file_path):
        preprocess_data(file_path)
    else:
        print(f"Error: {file_path} not found.")
