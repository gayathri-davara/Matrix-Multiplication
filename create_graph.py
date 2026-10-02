import matplotlib.pyplot as plt

implementations = [
    "Sequential",
    "OpenMP",
    "MPI",
    "CUDA Kernel"
]

execution_times = [
    300.472835,
    117.790996,
    210.911815,
    0.211245
]

plt.figure(figsize=(9, 5))

bars = plt.bar(implementations, execution_times)

plt.xlabel("Implementation")
plt.ylabel("Execution Time (seconds)")
plt.title("Matrix Multiplication - Execution Time Comparison")

for bar, value in zip(bars, execution_times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.6f}s",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "graphs/execution_time_comparison.png",
    dpi=300
)

plt.show()




# Speedup graph

implementations = [
    "OpenMP",
    "MPI",
    "CUDA Kernel"
]

speedups = [
    2.55,
    1.42,
    1422.87
]

plt.figure(figsize=(9, 5))

bars = plt.bar(implementations, speedups)

plt.xlabel("Implementation")
plt.ylabel("Speedup (×)")
plt.title("Observed Speedup Relative to Sequential")

for bar, value in zip(bars, speedups):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}×",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "graphs/speedup_comparison.png",
    dpi=300
)

plt.show()



# CPU and MPI execution time comparison

implementations = [
    "Sequential",
    "OpenMP",
    "MPI"
]

execution_times = [
    300.472835,
    117.790996,
    210.911815
]

plt.figure(figsize=(9, 5))

bars = plt.bar(implementations, execution_times)

plt.xlabel("Implementation")
plt.ylabel("Execution Time (seconds)")
plt.title("CPU and MPI Execution Time Comparison")

for bar, value in zip(bars, execution_times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}s",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "graphs/cpu_mpi_execution_time.png",
    dpi=300
)

plt.show()