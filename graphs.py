import os
import matplotlib.pyplot as plt

# Create output folder
output_dir = "poster_graphs"
os.makedirs(output_dir, exist_ok=True)

# Graph 1: Average Unsupported Claim Rate

experiments = ["Exp 1", "Exp 2", "Exp 3"]
unsupported_rates = [0.81, 0.735, 0.715]

plt.figure(figsize=(8, 5))
bars = plt.bar(experiments, unsupported_rates)
plt.title("Average Unsupported Claim Rate Across Experiments")
plt.xlabel("Experiment")
plt.ylabel("Average Unsupported Claim Rate")
plt.ylim(0, 1)

for bar, value in zip(bars, unsupported_rates):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "unsupported_rate_across_experiments.png"), dpi=300)
plt.close()

# Graph 2: Summary of dataset

items = ["Questions", "Documents", "Chunks"]
values = [100, 992, 1023]

plt.figure(figsize=(8, 5))
bars = plt.bar(items, values)
plt.title("Dataset and Preprocessing Summary")
plt.ylabel("Count")

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 10,
        str(value),
        ha="center",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dataset_summary.png"), dpi=300)
plt.close()

# Graph 3: Threshold and Retrieval Comparison

exp_names = ["Exp 1", "Exp 2", "Exp 3"]
top_k = [3, 3, 5]
supported_threshold = [0.75, 0.65, 0.65]
partial_threshold = [0.55, 0.45, 0.45]

plt.figure(figsize=(9, 5))
x = range(len(exp_names))

plt.plot(x, top_k, marker='o', label="Top-k")
plt.plot(x, supported_threshold, marker='s', label="Supported Threshold")
plt.plot(x, partial_threshold, marker='^', label="Partial Threshold")

plt.xticks(x, exp_names)
plt.title("Experiment Settings Comparison")
plt.xlabel("Experiment")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "experiment_settings_comparison.png"), dpi=300)
plt.close()

# Graph 4: Improvement in Unsupported Rate

improvement_labels = ["Exp1→Exp2", "Exp2→Exp3", "Exp1→Exp3"]
improvements = [
    unsupported_rates[0] - unsupported_rates[1],
    unsupported_rates[1] - unsupported_rates[2],
    unsupported_rates[0] - unsupported_rates[2]
]

plt.figure(figsize=(8, 5))
bars = plt.bar(improvement_labels, improvements)
plt.title("Reduction in Unsupported Claim Rate")
plt.ylabel("Improvement")

for bar, value in zip(bars, improvements):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.005,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "unsupported_rate_improvement.png"), dpi=300)
plt.close()

print("All poster graphs created successfully.")
print(f"Saved in folder: {os.path.abspath(output_dir)}")