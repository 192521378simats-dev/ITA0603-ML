from sklearn.tree import DecisionTreeClassifier

# Dataset: Income, Debt, Payment History
X = [[25000, 20000, 0], [50000, 10000, 1],
     [30000, 25000, 0], [80000, 15000, 1],
     [60000, 12000, 1], [20000, 18000, 0]]

y = ['Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor']

# Train model
model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)

# Predict credit score
result = model.predict([[55000, 12000, 1]])
print("Credit Score:", result[0])