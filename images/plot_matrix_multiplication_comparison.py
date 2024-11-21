import matplotlib.pyplot as plt

# Data
sizes = [64, 128, 256, 512, 1024]
fast_times = [0.00276, 0.01316, 0.08600, 1.22291, 7.77354]
gpu_times = [0.00537, 0.01160, 0.04131, 0.21219, 0.79437]

# Plot with customizations
plt.figure(figsize=(10, 6))
plt.plot(sizes, fast_times, marker="o", color="blue", label="Fast CPU")
plt.plot(sizes, gpu_times, marker="o", color="red", label="GPU")

# Labels and title
plt.xlabel("Input Size")
plt.ylabel("Runtime (seconds)")
plt.title("Matrix Multiplication Runtime Comparison: Fast CPU vs GPU Implementation")
plt.xlim(0, 1100)  # Limit x-axis from 0 to 1000
plt.ylim(0, 10)  # Limit y-axis from 0 to 10
plt.grid(True, which="both", linestyle="--", linewidth=0.75)
plt.legend()
plt.show()
