import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Reading the data
data = pd.read_csv("/Users/utkarshtripathi/Desktop/bird strike /Bird Strikes.csv")

# Preprocessing: Drop unnecessary columns and handle missing values
columns_to_remove = ['Record ID', 'Aircraft Type', 'Remains of wildlife sent to Smithsonian', 
                     'Remarks', 'Pilot warned of birds or wildlife?', 'Effect Impact to flight', 
                     'Cost Total ', 'Number of people injured', 'Remains of wildlife collected?',
                     'Wildlife Number struck', 'Wildlife Number Struck Actual']
data = data.drop(columns=columns_to_remove)
data.dropna(inplace=True)

# Identify features and target variable
target_column = 'Effect: Indicated Damage'
features = data.columns.drop(target_column)

# Separate features and target variable
X = data[features]
y = data[target_column]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ]
)

# Create a complete pipeline with preprocessing and classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the trained model pipeline
joblib.dump(model_pipeline, 'model/bird_strike_model.pkl')
