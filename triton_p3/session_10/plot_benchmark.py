import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

N = [512, 1024, 2048, 4096]
flash = [0.027, 0.043, 0.082, 0.159]
standard = [0.034, 0.057, 0.179, 1.006]
speedup = [s / f for f, s in zip(flash, standard)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("FlashAttention vs Standard Attention (Causal, D=64)", fontsize=14, fontweight="bold")

# latency plot
ax1.plot(N, standard, "o-", color="#e05252", linewidth=2, markersize=7, label="Standard Attention")
ax1.plot(N, flash, "o-", color="#4a9eff", linewidth=2, markersize=7, label="FlashAttention (ours)")
ax1.set_xlabel("Sequence Length (N)")
ax1.set_ylabel("Latency (ms)")
ax1.set_title("Latency")
ax1.legend()
ax1.set_xticks(N)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# speedup plot
bars = ax2.bar(N, speedup, width=250, color="#4a9eff", alpha=0.85)
ax2.set_xlabel("Sequence Length (N)")
ax2.set_ylabel("Speedup (x)")
ax2.set_title("Speedup over Standard Attention")
ax2.set_xticks(N)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
for bar, s in zip(bars, speedup):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f"{s:.2f}x", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("benchmark.png", dpi=150, bbox_inches="tight")
print("saved to benchmark.png")
