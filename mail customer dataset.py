import pandas as pd
df = pd.read_csv('mail_customer_dataset.csv') # Replace 'customer_data.csv' with your dataset filename
df.fillna(df.mean(), inplace=True) # Example: fill with mean for numerical columns
df = pd.get_dummies(df, columns=['Gender', 'Region']) 
    # Example: one-hot encode 'Gender' and 'Region'
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']]) # Example: scale 'Age' and 'Income'    
from sklearn.model_selection import train_test_split
X = df.drop('Target_Variable', axis=1) # Replace 'Target_Variable'
y = df['Target_Variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier # Example: Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

