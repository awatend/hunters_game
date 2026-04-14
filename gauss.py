import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(0, 300, 10, dtype=np.float32)

# Compute the function
y = np.exp(-(x / 50.0) ** 2).astype(np.float32)

# Plot
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("exp(-(x/50)^2)")
plt.title("Gaussian Decay Function")
plt.grid()

plt.show()