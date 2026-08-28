import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def mutate(ray: np.ndarray) -> np.ndarray:
    """Flip a random subset of bits in a binary feature-selection vector."""
    mutation = np.random.randint(0, 2, size=ray.shape)
    return ray ^ mutation


def fitness_function(features: np.ndarray, X, y) -> float:
    """
    Evaluate a candidate feature subset by training a fast XGBoost model
    and returning negative validation MSE (higher fitness = lower error).
    """
    selected_indices = np.where(features == 1)[0]
    if len(selected_indices) == 0:
        return -np.inf  # an empty feature subset is never viable

    X_selected = X.iloc[:, selected_indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    val_mse = mean_squared_error(y_val, preds)

    return -val_mse


def manta_ray_foraging_optimization(
    X, y, num_iterations: int = 10, num_manta_rays: int = 10, verbose: bool = True
) -> np.ndarray:
    """
    Run MRFO-based feature selection, returning a binary vector indicating
    which columns of X were selected.
    """
    num_features = X.shape[1]
    manta_rays = np.random.randint(2, size=(num_manta_rays, num_features))

    for iteration in range(num_iterations):
        fitness_scores = [fitness_function(ray, X, y) for ray in manta_rays]

        best_indices = np.argsort(fitness_scores)[::-1]  # descending: best fitness first
        best_rays = [manta_rays[i] for i in best_indices[: num_manta_rays // 2]]

        manta_rays = np.array(
            best_rays
            + [
                mutate(np.copy(best_rays[np.random.randint(len(best_rays))]))
                for _ in range(num_manta_rays // 2)
            ]
        )

        if verbose:
            print(f"Iteration {iteration + 1}/{num_iterations} — best fitness: {max(fitness_scores):.2f}")

    final_scores = [fitness_function(ray, X, y) for ray in manta_rays]
    best_manta_ray = manta_rays[np.argmax(final_scores)]
    return best_manta_ray