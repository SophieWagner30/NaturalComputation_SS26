import numpy as np
import pandas as pd

def generate_hard(n_samples=2000):
    X = np.random.uniform(-2, 2, (n_samples, 10))
    
    y = (
        np.sin(X[:, 0]) +
        X[:, 1]**2 -
        np.cos(X[:, 2]) +
        X[:, 3] * X[:, 4]
    )
    
    y = (y > np.median(y)).astype(int)
    
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(10)])
    df['target'] = y
    
    return df


df = generate_hard()
df.to_csv("nonlinear_dataset.csv", index=False)
print("Saved nonlinear_dataset.csv")