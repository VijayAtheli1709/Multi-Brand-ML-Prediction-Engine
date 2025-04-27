import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataframe_vw = pd.read_csv('vw.csv')
dataframe_audi = pd.read_csv('audi.csv')
dataframe_bmw = pd.read_csv('bmw.csv')
dataframe_ford = pd.read_csv('ford.csv')
dataframe_hyundai = pd.read_csv('hyundai.csv')
dataframe_mercedes = pd.read_csv('mercedes.csv')
dataframe_skoda = pd.read_csv('skoda.csv')
dataframe_toyota = pd.read_csv('toyota.csv')
dataframe_vauxhall = pd.read_csv('vauxhall.csv')

merged_dataframe = pd.concat([dataframe_vw, dataframe_audi,dataframe_bmw,dataframe_ford,dataframe_hyundai,dataframe_mercedes,dataframe_skoda,dataframe_toyota,dataframe_vauxhall])
merged_dataframe.info
merged_dataframe.to_excel('merged_cars_data.xlsx', index=False)  
merged_dataframe.columns

# Load the Excel file
df = pd.read_excel('merged_cars_data.xlsx')

# Drop the incorrect tax column
df = df.drop(columns=['tax(£)'])

# Fill missing 'tax' values using median grouped by 'engineSize' and 'fuelType'
print(df['tax'].isnull().sum())  # Before filling
df['tax'] = df.groupby(['engineSize', 'fuelType'])['tax'].transform(
    lambda x: x.fillna(x.median())
)
print(df['tax'].isnull().sum())  # After filling

# Drop rows where 'tax' is still missing
df = df.dropna(subset=['tax'])

# Strip whitespace from 'model' column
df['model'] = df['model'].str.strip()

# Filter 'year' column to keep values only between 1999 and 2025
df = df[(df['year'] >= 1999) & (df['year'] <= 2025)]

# Print unique values in each column
for col in df.columns:
    print(f"{col}:\n{df[col].unique()}\n")

# Check final dataset info
df.info()

# Save cleaned data to Excel
#df.to_excel("Final_dataset.xlsx", index=False)


# Select only numerical columns
numeric_df = df.select_dtypes(include='number')

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap focused on correlation with 'price'
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid", font_scale=1.1)

# Create a color palette from red (negative) to green (positive)
sns.heatmap(
    correlation_matrix[['price']].sort_values(by='price', ascending=False),
    annot=True,
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Correlation of 'price' with Other Numerical Features", fontsize=14)
plt.tight_layout()
plt.show()