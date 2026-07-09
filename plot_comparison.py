import pandas as pd
import matplotlib.pyplot as plt

def plot_final_comparison():
    files = {
        "MLP (Reactive)": "mlp_model_log.csv",
        "GRU (Memory)": "gru_final_log.csv",
        "Residual (Reflexes)": "res_mlp_v1_log.csv"
    }
    
    plt.figure(figsize=(12, 7))
    for label, name in files.items():
        df = pd.read_csv(f"logs/{name}")
        plt.plot(df['gen'], df['best_fitness'], label=label, linewidth=2)

    plt.title("Comparison: MLP vs GRU vs Residual", fontsize=14)
    plt.xlabel("Generations", fontsize=12)
    plt.ylabel("Best Fitness", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparative_efsim.png")
    print("Comparative! Check 'comparative_efsim.png'")
    plt.show()

if __name__ == "__main__":
    plot_final_comparison()

