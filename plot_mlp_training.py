import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_residual_only():
    log_path = Path("logs/res_mlp_v1_log.csv")
    
    if not log_path.exists():
        print(f"Error: File not found {log_path}")
        return

    df = pd.read_csv(log_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- AXIS 1: FITNESS (Red) ---
    ax1.set_xlabel('Generations (Time)')
    ax1.set_ylabel('Fitness Scale', color='tab:red', fontsize=12)
    
    # Plot Best and Mean
    lns1 = ax1.plot(df['gen'], df['best_fitness'], color='tab:red', 
                    label='Best Fitness', linewidth=2)
    lns2 = ax1.plot(df['gen'], df['mean_fitness'], color='tab:red', alpha=0.4, 
                    linestyle='--', label='Population Mean')
    
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    # --- AXIS 2: SUCCESS (Blue) ---
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Success Rate (0.0 to 1.0)', color='tab:blue', fontsize=12)
    
    lns3 = ax2.plot(df['gen'], df['best_success'], color='tab:blue', 
                    label='Success Rate', linewidth=2)
    
    ax2.set_ylim(-0.05, 1.05) # Limit from 0 to 100%
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # --- Unified Legend ---
    # Combine labels from both axes into one legend
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', frameon=True, shadow=True)

    plt.title("Residual MLP Training\n(Direct reflexes + correction layer)", fontsize=14)
    fig.tight_layout()
    
    output_img = "residual_analysis_final.png"
    plt.savefig(output_img)
    print(f"EFSIM: Plot generated successfully: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_residual_only()

