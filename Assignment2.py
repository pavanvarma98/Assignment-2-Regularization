import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the data set
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name = "target")

print("=" * 30)
print("X shape", X.shape)
print("y shape", y.shape)
print(X.head())
print("=" * 30)

# train test split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#feature scaling this is important for regularization

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#linear regression model

linear = LinearRegression()
linear.fit(X_train_scaled, y_train) 

y_train_pred = linear.predict(X_train_scaled)
y_test_pred = linear.predict(X_test_scaled)

lin_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
lin_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
lin_test_r2 = r2_score(y_test, y_test_pred)

print("=" * 30)
print("\n--- Linear Regression ---")
print("Train RMSE:", round(lin_train_rmse, 4))
print("Test RMSE:", round(lin_test_rmse, 4))
print("Test R2:", round(lin_test_r2, 4))
print("=" * 30)

#regularization tuning for ridge, lasso, elasticnet

alphas = np.logspace(-4, 3, 30)

ridge_train_rmse_list, ridge_test_rmse_list = [], []
lasso_train_rmse_list, lasso_test_rmse_list = [], []
enet_train_rmse_list, enet_test_rmse_list = [], []

ridge_coefs = []
lasso_coefs = []
enet_coefs = []

# elastic net tuning

l1_ratio = 0.5  # TODO: try 0.2, 0.5, 0.8 and choose best later

best = {
    "ridge": {"alpha": None, "test_rmse": float("inf"), "model": None},
    "lasso": {"alpha": None, "test_rmse": float("inf"), "model": None},
    "enet": {"alpha": None, "test_rmse": float("inf"), "model": None},
}

for a in alphas:
    # Ridge
    ridge = Ridge(alpha=a, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    ridge_train_pred = ridge.predict(X_train_scaled)
    ridge_test_pred = ridge.predict(X_test_scaled)

    ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_pred))

    ridge_train_rmse_list.append(ridge_train_rmse)
    ridge_test_rmse_list.append(ridge_test_rmse)
    ridge_coefs.append(ridge.coef_)

    if ridge_test_rmse < best["ridge"]["test_rmse"]:
        best["ridge"] = {"alpha": a, "test_rmse": ridge_test_rmse, "model": ridge}

    # Lasso
    lasso = Lasso(alpha=a, max_iter=20000, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    lasso_train_pred = lasso.predict(X_train_scaled)
    lasso_test_pred = lasso.predict(X_test_scaled)

    lasso_train_rmse = np.sqrt(mean_squared_error(y_train, lasso_train_pred))
    lasso_test_rmse = np.sqrt(mean_squared_error(y_test, lasso_test_pred))

    lasso_train_rmse_list.append(lasso_train_rmse)
    lasso_test_rmse_list.append(lasso_test_rmse)
    lasso_coefs.append(lasso.coef_)

    if lasso_test_rmse < best["lasso"]["test_rmse"]:
        best["lasso"] = {"alpha": a, "test_rmse": lasso_test_rmse, "model": lasso}

    # Elastic Net
    enet = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=20000, random_state=42)
    enet.fit(X_train_scaled, y_train)
    enet_train_pred = enet.predict(X_train_scaled)
    enet_test_pred = enet.predict(X_test_scaled)

    enet_train_rmse = np.sqrt(mean_squared_error(y_train, enet_train_pred))
    enet_test_rmse = np.sqrt(mean_squared_error(y_test, enet_test_pred))

    enet_train_rmse_list.append(enet_train_rmse)
    enet_test_rmse_list.append(enet_test_rmse)
    enet_coefs.append(enet.coef_)

    if enet_test_rmse < best["enet"]["test_rmse"]:
        best["enet"] = {"alpha": a, "test_rmse": enet_test_rmse, "model": enet}


print("\n--- Best tuned models (by lowest Test RMSE) ---")
print("Ridge  best alpha:", best["ridge"]["alpha"], "Test RMSE:", round(best["ridge"]["test_rmse"], 4))
print("Lasso  best alpha:", best["lasso"]["alpha"], "Test RMSE:", round(best["lasso"]["test_rmse"], 4))
print("Enet   best alpha:", best["enet"]["alpha"],  "Test RMSE:", round(best["enet"]["test_rmse"], 4), "l1_ratio:", l1_ratio)


# -------------------------
# 6) Final evaluation table (simple)
# -------------------------
# TODO: Add R2 for best ridge/lasso/enet if your assignment needs it

# -------------------------
# 7) VISUALIZATION SECTION (ALL PLOTS AT THE END)
# -------------------------

# Plot 1: Train vs Test RMSE vs alpha (Ridge/Lasso/ElasticNet)
plt.figure(figsize=(8, 5))
plt.plot(alphas, ridge_train_rmse_list, label="Ridge Train RMSE")
plt.plot(alphas, ridge_test_rmse_list, label="Ridge Test RMSE")
plt.xscale("log")
plt.xlabel("alpha (log scale)")
plt.ylabel("RMSE")
plt.title("Ridge: Train vs Test RMSE vs alpha")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(alphas, lasso_train_rmse_list, label="Lasso Train RMSE")
plt.plot(alphas, lasso_test_rmse_list, label="Lasso Test RMSE")
plt.xscale("log")
plt.xlabel("alpha (log scale)")
plt.ylabel("RMSE")
plt.title("Lasso: Train vs Test RMSE vs alpha")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(alphas, enet_train_rmse_list, label="ElasticNet Train RMSE")
plt.plot(alphas, enet_test_rmse_list, label="ElasticNet Test RMSE")
plt.xscale("log")
plt.xlabel("alpha (log scale)")
plt.ylabel("RMSE")
plt.title(f"ElasticNet (l1_ratio={l1_ratio}): Train vs Test RMSE vs alpha")
plt.legend()
plt.show()


# Plot 2: Coefficient shrinkage paths
ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)
enet_coefs = np.array(enet_coefs)

plt.figure(figsize=(8, 5))
for j in range(ridge_coefs.shape[1]):
    plt.plot(alphas, ridge_coefs[:, j])
plt.xscale("log")
plt.xlabel("alpha (log scale)")
plt.ylabel("Coefficient value")
plt.title("Ridge: Coefficient paths (shrinkage)")
plt.show()

plt.figure(figsize=(8, 5))
for j in range(lasso_coefs.shape[1]):
    plt.plot(alphas, lasso_coefs[:, j])
    plt.xscale("log")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("Coefficient value")
    plt.title("Lasso: Coefficient paths (some go to 0)")
    plt.show()

plt.figure(figsize=(8, 5))
for j in range(enet_coefs.shape[1]):
    plt.plot(alphas, enet_coefs[:, j])
    plt.xscale("log")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("Coefficient value")
    plt.title(f"ElasticNet (l1_ratio={l1_ratio}): Coefficient paths")
    plt.show()
