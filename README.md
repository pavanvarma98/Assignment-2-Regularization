# Assignment-2-Regularization
# Assignment 2 – Regularization Techniques

## Objective
To implement and compare Linear Regression, Ridge, Lasso, and Elastic Net models
using a real-world dataset and analyze the effect of regularization.

## Dataset
California Housing dataset from scikit-learn  
Features: 8 numerical features  
Target: Median house value

## Steps Performed
1. Loaded and explored the dataset
2. Performed train-test split
3. Applied feature scaling using StandardScaler
4. Built a baseline Linear Regression model
5. Tuned Ridge, Lasso, and Elastic Net models using different alpha values
6. Evaluated models using RMSE and R²
7. Visualized:
   - Train vs Test RMSE vs regularization strength
   - Coefficient shrinkage paths

## Conclusion
Regularization improved model generalization by reducing overfitting.
Lasso performed feature selection by shrinking some coefficients to zero,
while Ridge shrunk coefficients without eliminating features.
Elastic Net balanced both approaches.

## How to Run
```bash
python Assignment2.py
