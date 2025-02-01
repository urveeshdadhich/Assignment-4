import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
X = pd.read_csv('logisticX.csv', header=None).values
y = pd.read_csv('logisticY.csv', header=None).values.flatten()

# Normalize features
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

class LogisticRegression:
    def __init__(self, lr=0.1, max_iters=1000, tol=1e-4):
        self.theta = None
        self.costs = []
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.max_iters):
            z = X @ self.theta
            h = self._sigmoid(z)
            gradient = X.T @ (h - y) / y.size
            self.theta -= self.lr * gradient
            
            cost = -np.mean(y * np.log(h + 1e-8) + (1-y) * np.log(1-h + 1e-8))
            self.costs.append(cost)
            
            if i > 0 and abs(self.costs[-1] - self.costs[-2]) < self.tol:
                break

# Q1: Train with η=0.1
model_01 = LogisticRegression(lr=0.1)
model_01.fit(X_normalized, y)
print(f"Final theta: {model_01.theta}")
print(f"Final cost: {model_01.costs[-1]:.4f}")

# Q2: Plot cost vs iterations
plt.figure(figsize=(10,6))
plt.plot(model_01.costs, 'b-')
plt.title('Cost Function vs Iterations (η=0.1)')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

# Q3: Plot decision boundary
plt.figure(figsize=(10,6))
plt.scatter(X_normalized[y==0][:,0], X_normalized[y==0][:,1], color='blue', label='Class 0')
plt.scatter(X_normalized[y==1][:,0], X_normalized[y==1][:,1], color='red', label='Class 1')

x_bound = np.array([X_normalized[:,0].min(), X_normalized[:,0].max()])
y_bound = (-model_01.theta[0] - model_01.theta[1]*x_bound) / model_01.theta[2]
plt.plot(x_bound, y_bound, 'k--', label='Decision Boundary')

plt.title('Classification Results')
plt.xlabel('Feature 1 (normalized)')
plt.ylabel('Feature 2 (normalized)')
plt.legend()
plt.grid(True)
plt.show()

# Q4: Compare learning rates
model_5 = LogisticRegression(lr=5, max_iters=100)
model_5.fit(X_normalized, y)

plt.figure(figsize=(10,6))
plt.plot(model_01.costs[:100], 'b-', label='η=0.1')
plt.plot(model_5.costs, 'r-', label='η=5')
plt.title('Learning Rate Comparison')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.show()

# Q5: Evaluation metrics
def predict(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]
    return (X @ theta >= 0).astype(int)

y_pred = predict(X_normalized, model_01.theta)

conf_matrix = np.zeros((2,2))
conf_matrix[0,0] = np.sum((y==0) & (y_pred==0))  # TN
conf_matrix[0,1] = np.sum((y==0) & (y_pred==1))  # FP
conf_matrix[1,0] = np.sum((y==1) & (y_pred==0))  # FN
conf_matrix[1,1] = np.sum((y==1) & (y_pred==1))  # TP

accuracy = (conf_matrix[0,0] + conf_matrix[1,1]) / y.size
precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1])
recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
