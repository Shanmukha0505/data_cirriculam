
import pandas as pd

# Load and prepare the data files
df_restaurant = pd.read_excel('Restaurant-inspection-Results.xlsx')
df_yelp = pd.read_excel('yelp-Data.xlsx')

# Create Final_Inspection_Data.csv (cleaned restaurant data)
df_final = df_restaurant.copy()

# Clean the data
threshold = len(df_final) * 0.1
df_final = df_final.dropna(axis=1, thresh=threshold)

# Fix dates
for col in df_final.select_dtypes(include=['datetime64[ns]']):
    df_final[col] = df_final[col].mask(df_final[col] == pd.Timestamp('1900-01-01'))

# Format phone numbers
def format_phone(x):
    if pd.isna(x):
        return ''
    if isinstance(x, (int, float)):
        return '{:.0f}'.format(x)
    return str(x)

if 'PHONE' in df_final.columns:
    df_final['PHONE'] = df_final['PHONE'].apply(format_phone)

# Format ZIP codes
def format_zip(z):
    if pd.isna(z):
        return ''
    try:
        return str(int(z)).zfill(5)
    except:
        return str(z)

if 'ZIPCODE' in df_final.columns:
    df_final['ZIPCODE'] = df_final['ZIPCODE'].apply(format_zip)

df_final = df_final.drop_duplicates().reset_index(drop=True)

# Save Final_Inspection_Data.csv
df_final.to_csv('data_curriculum/data/Final_Inspection_Data.csv', index=False)
print(f"Created: Final_Inspection_Data.csv with {df_final.shape[0]} rows, {df_final.shape[1]} columns")

# Create RecentInspDate.csv (filter for most recent inspections per restaurant)
if 'INSPECTION DATE' in df_final.columns and 'CAMIS' in df_final.columns:
    df_recent = df_final.sort_values('INSPECTION DATE', ascending=False)
    df_recent = df_recent.groupby('CAMIS').first().reset_index()
    df_recent.to_csv('data_curriculum/data/RecentInspDate.csv', index=False)
    print(f"Created: RecentInspDate.csv with {df_recent.shape[0]} rows, {df_recent.shape[1]} columns")
else:
    print("Warning: Could not create RecentInspDate.csv - missing required columns")
