import pandas as pd
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_excel('Final_dataset.xlsx')

df.columns

df['Brand'].unique()

VW_df = df.loc[df['Brand']=='VW']
Hyundai_df = df.loc[df['Brand']=='Hyundai']
Skoda_df = df.loc[df['Brand']=='Skoda']

print(VW_df)

#VW
VW_df.info()
for col in VW_df.columns:
    print(f"{col}:\n{VW_df[col].unique()}\n")
#correlation heatmap
numeric_cols = VW_df.select_dtypes(include=['int64', 'float64']).columns
correlation = VW_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for VW Numeric Features')
plt.tight_layout()
plt.show()

#scatterplot
plt.figure(figsize=(10, 6))
sns.regplot(x='mileage', y='price', data=VW_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('VW Price vs Mileage')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

#Linear Regression Model 
VW_df = VW_df.drop(columns=[ "Brand"])
VW_df_dummy=pd.get_dummies(VW_df,drop_first=True)

y=VW_df_dummy['price']
x=VW_df_dummy.drop(columns=['price'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

##building the linear regression model
##1. import #2. initialize #3 train #4. evauluate 

lm=LinearRegression()#initialize
lm.fit(x_train, y_train)#train
y_pred = lm.predict(x_test)

# Calculate evaluation metrics:
# Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error
import numpy as np
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse1) #2551.2092

#Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor

# Split data into train and test (70/30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Initialize and train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(x_train, y_train)

# Predict on test data
y_pred = dt_model.predict(x_test)

# Evaluation metrics
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse2) #2133.6481

# Price by fuel type
sns.boxplot(x='fuelType', y='price', data= VW_df)
plt.show()

#Average Price over years 
VW_df.groupby('year')['price'].mean().plot()
plt.show()

#The Decision Tree model outperforms Linear Regression for VW car price 
#prediction with a lower RMSE of 2133.65 compared to 2551.21. It captures non-linear 
#patterns and interactions between features like mileage, engine size, and year.
#Unlike Linear Regression, it handles sharp value changes and threshold effects more effectively..


#Hyundai
print(Hyundai_df)

#VW
Hyundai_df.info()

# change the dataset name in the first line and dataset name in 2n line
for col in Hyundai_df.columns:
    print(f"{col}:\n{Hyundai_df[col].unique()}\n")

#correlation heatmap
numeric_cols = Hyundai_df.select_dtypes(include=['int64', 'float64']).columns
correlation = Hyundai_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Hyundai Numeric Features')
plt.tight_layout()
plt.show()

#scatterplot
plt.figure(figsize=(10, 6))
sns.regplot(x='mileage', y='price', data=Hyundai_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Hyundai Price vs Mileage')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

#Linear Regression Model 
Hyundai_df = Hyundai_df.drop(columns=[ "Brand"])
Hyundai_df_dummy=pd.get_dummies(Hyundai_df,drop_first=True)

y=Hyundai_df_dummy['price']
x=Hyundai_df_dummy.drop(columns=['price'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

##building the linear regression model
##1. import #2. initialize #3 train #4. evauluate 
lm=LinearRegression()#initialize
lm.fit(x_train, y_train)#train
y_pred = lm.predict(x_test)

# Calculate evaluation metrics:
# Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error
import numpy as np
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse1) #2041.1300

#Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor

# Split data into train and test (70/30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Initialize and train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(x_train, y_train)

# Predict on test data
y_pred = dt_model.predict(x_test)

# Evaluation metrics
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse2) #2698.2224

# Price by fuel type
sns.boxplot(x='fuelType', y='price', data= Hyundai_df)
plt.show()

#Average Price over years 
Hyundai_df.groupby('year')['price'].mean().plot()
plt.show()

#Linear Regression is performing better because it has a lower RMSE, 
#meaning its predictions are closer to the actual prices on average. 
#It assumes a linear relationship between features and price.

#Skoda
print(Skoda_df)

#Skoda
Skoda_df.info()

# change the dataset name in the first line and dataset name in 2n line
for col in Skoda_df.columns:
    print(f"{col}:\n{Skoda_df[col].unique()}\n")

#correlation heatmap
numeric_cols = Skoda_df.select_dtypes(include=['int64', 'float64']).columns
correlation = Skoda_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Skoda Numeric Features')
plt.tight_layout()
plt.show()

#scatterplot
plt.figure(figsize=(10, 6))
sns.regplot(x='mileage', y='price', data=Skoda_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Skoda Price vs Mileage')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

#Multiple Linear Regression Model 
Skoda_df = Skoda_df.drop(columns=[ "Brand"])
Skoda_df_dummy=pd.get_dummies(Skoda_df,drop_first=True)

y=Skoda_df_dummy['price']
x=Skoda_df_dummy.drop(columns=['price'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

##building the linear regression model
##1. import #2. initialize #3 train #4. evauluate 
lm=LinearRegression()#initialize
lm.fit(x_train, y_train)#train
y_pred = lm.predict(x_test)

# Calculate evaluation metrics:
# Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error
import numpy as np
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse1) #1828.4836

#Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor

# Split data into train and test (70/30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Initialize and train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(x_train, y_train)

# Predict on test data
y_pred = dt_model.predict(x_test)

# Evaluation metrics
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse2) #1808.7758

# Price by fuel type
sns.boxplot(x='fuelType', y='price', data= Skoda_df)
plt.show()

#Average Price over years 
Skoda_df.groupby('year')['price'].mean().plot()
plt.show()

#Decision Tree is the better model here, although the difference is small. 
#It performs better likely because it can model non-linear interactions between 
#features like mileage, year, and model more effectively than Linear Regression.


#Load Packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

BMW_df = df.loc[df['Brand']=='BMW']
Ford_df = df.loc[df['Brand']=='Ford']
Toyota_df = df.loc[df['Brand']=='Toyota']


#FORD
Ford_df.info()

for col in Ford_df.columns:
    print(f"{col}:\n{Ford_df[col].unique()}\n")
#Multiple Linear Regression Model 
Ford_df = Ford_df.drop(columns=[ "Brand"])
Ford_df_dummy=pd.get_dummies(Ford_df,drop_first=True)

y=Ford_df_dummy['price']
x=Ford_df_dummy.drop(columns=['price'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3
                                               ,random_state=1)

##building the linear regression model
##1. import #2. initialize #3 train #4. evauluate 

lm=LinearRegression()#initialize

lm.fit(x_train, y_train)#train

y_pred = lm.predict(x_test)

# Calculate evaluation metrics:
# Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error
import numpy as np

rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse1) #1874.4670

#Decision Tree Regressor 

from sklearn.tree import DecisionTreeRegressor

# Split data into train and test (70/30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Initialize and train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(x_train, y_train)

# Predict on test data
y_pred = dt_model.predict(x_test)

# Evaluation metrics
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse2) #1617.2631959627247

# Price by fuel type
sns.boxplot(x='fuelType', y='price', data= Ford_df)
plt.show()

#Average Price over years 
Ford_df.groupby('year')['price'].mean().plot()
plt.show()

#correlation heatmap
numeric_cols = Ford_df.select_dtypes(include=['int64', 'float64']).columns
correlation = Ford_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Hyundai Numeric Features')
plt.tight_layout()
plt.show()

#Decision Tree Regressor performs slightly better likely because it captures non-linear patterns
#and interactions between features like mileage, engine size, and year.
#Decision tree also has a lower rmse than the multiple linear regression model 

#BMW

BMW_df.info()

# change the dataset name in the first line and dataset name in 2n line
for col in BMW_df.columns:
    print(f"{col}:\n{BMW_df[col].unique()}\n")

BMW_df = BMW_df.drop(columns=[ "Brand"])

BMW_df_dummy=pd.get_dummies(BMW_df,drop_first=True)

#Multiple Linear Regression
y=BMW_df_dummy['price']
x=BMW_df_dummy.drop(columns=['price'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3
                                               ,random_state=1)

##building the linear regression model
##1. import #2. initialize #3 train #4. evauluate 

lm=LinearRegression()#initialize

lm.fit(x_train, y_train)#train

y_pred = lm.predict(x_test)

# Calculate evaluation metrics:
# Root Mean Squared Error (RMSE)


from sklearn.metrics import mean_squared_error
import numpy as np

rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse1) #4092.264994881996


#Random Forest 

from sklearn.ensemble import RandomForestRegressor
y=BMW_df_dummy['price']
x=BMW_df_dummy.drop(columns=['price'])

# Step 5: Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Step 6: Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Step 7: Make predictions
y_pred = rf_model.predict(x_test)

# Step 8: Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse) #2528.5161607922405

#Average Price by Top 10 Models
top_models = BMW_df['model'].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.barplot(data=df[df['model'].isin(top_models)], x='model', y='price', estimator='mean')
plt.title('Average Price by Top 10 Models')
plt.xticks(rotation=45)
plt.show()

#Average Price over years 
BMW_df.groupby('year')['price'].mean().plot()
plt.show()

#correlation heatmap
numeric_cols = BMW_df.select_dtypes(include=['int64', 'float64']).columns
correlation = BMW_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Hyundai Numeric Features')
plt.tight_layout()
plt.show()

#Random Forest tregressor is a better model to predict prices as 
#Lower average prediction error (RMSE = 2528.5161)
#Better handling of real-world, non-linear patterns in your data

#Toyota 

Toyota_df.to_excel('Toyota.xlsx')

#Decision Tree Regresosor 
from sklearn.tree import DecisionTreeRegressor

Toyota_df = Toyota_df.drop(columns=[ "Brand"])

Toyota_df_dummy=pd.get_dummies(Toyota_df,drop_first=True)

y=Toyota_df_dummy['price']
x=Toyota_df_dummy.drop(columns=['price'])


# Split data into train and test (70/30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Initialize and train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(x_train, y_train)

# Predict on test data
y_pred = dt_model.predict(x_test)

# Evaluation metrics
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse2) #1478.831127436685

# Random trees regressor 

from sklearn.ensemble import RandomForestRegressor
y=Toyota_df_dummy['price']
x=Toyota_df_dummy.drop(columns=['price'])


# Step 5: Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Step 6: Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Step 7: Make predictions
y_pred = rf_model.predict(x_test)

# Step 8: Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse) #1162.9560729982531

plt.figure(figsize=(8, 5))
sns.boxplot(x='transmission', y='price', data=Toyota_df)
plt.title("Price by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Price")
plt.show()

#Average Price over years 
BMW_df.groupby('year')['price'].mean().plot()
plt.show()

#correlation heatmap
numeric_cols = Toyota_df.select_dtypes(include=['int64', 'float64']).columns
correlation = Toyota_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Hyundai Numeric Features')
plt.tight_layout()
plt.show()

#Random Forest Regressor is better as more accurate, stable, and less likely
#to overfit compared to a single decision tree — especially in real-world 
#datasets like your car pricing model. It also has a lower RMSE 


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Brands to Analyze
brands_to_check = ['Mercedes', 'Audi', 'Vauxhall']

# Filter brands
brands_to_check = ['Mercedes', 'Audi', 'Vauxhall']
df_selected_brands = df[df['Brand'].isin(brands_to_check)]

for brand in brands_to_check:
    print(f"\n===== {brand} Analysis =====")
    
    brand_df = df[df['Brand'] == brand].copy()
    brand_df.drop(columns=['Brand'], inplace=True)
    
    # Dummy encoding for categorical variables
    brand_df_dummy = pd.get_dummies(brand_df, drop_first=True)
    
    # Define features and target
    y = brand_df_dummy['price']
    X = brand_df_dummy.drop(columns=['price'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    # Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=1)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    
    # Print RMSE results
    print(f"Linear Regression RMSE: {rmse_lr:.2f}")
    print(f"Decision Tree RMSE: {rmse_dt:.2f}")
    
    # Determine better model
    better_model = "Decision Tree" if rmse_dt < rmse_lr else "Linear Regression"
    print(f"Better Model: {better_model}")
    brand_df.to_excel(f"{brand}.xlsx", index=False)

    # Scatterplot: Price vs Mileage
    if 'mileage' in brand_df.columns and 'price' in brand_df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(x='mileage', y='price', data=brand_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f'{brand} Price vs Mileage')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.show()
    
    # Correlation heatmap
    numeric_cols = brand_df.select_dtypes(include=['int64', 'float64']).columns
    correlation = brand_df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Heatmap for {brand}')
    plt.tight_layout()
    plt.show()
    
    # Boxplot: Price by Fuel Type
    if 'fuelType' in brand_df.columns and 'price' in brand_df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='fuelType', y='price', data=brand_df)
        plt.title(f'{brand} Price by Fuel Type')
        plt.tight_layout()
        plt.show()
    
    # Line Plot: Average Price Over Years
    if 'year' in brand_df.columns and 'price' in brand_df.columns:
        plt.figure(figsize=(8, 5))
        brand_df.groupby('year')['price'].mean().plot(marker='o')
        plt.title(f'{brand} - Average Price Over Years')
        plt.xlabel('Year')
        plt.ylabel('Average Price')
        plt.tight_layout()
        plt.show()
        
'''Conclusion:
Across 'Mercedes', 'Audi', 'Vauxhall' brands, Decision Tree Regressor was the superior model. 
It better captured the non-linear and interaction effects in the data, especially 
for brands like Mercedes and Audi with diverse lineups. 
The combination of modeling and visual diagnostics provides a robust approach 
to understanding and forecasting vehicle pricing.
'''
