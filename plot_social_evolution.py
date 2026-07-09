import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_evolution_results(log_path, leg_loc): 
    # 1. Cargar datos
    if not os.path.exists(log_path):
        print(f"Error: File '{log_path}' not found.")
        return

    try:
        # Algunos CSVs pueden tener espacios después de las comas, los limpiamos
        df = pd.read_csv(log_path, skipinitialspace=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. TRADUCTOR UNIVERSAL DE COLUMNAS (Parcha todas tus versiones)
    column_map = {
        'best': 'best_fitness',
        'best_fitness': 'best_fitness',
        'avg': 'avg_fitness',
        'avg_fitness': 'avg_fitness',
        'mean_fitness': 'avg_fitness',    
        'time': 'time',
        'gen_time': 'time'              
    }
    df = df.rename(columns=column_map)

    # Verificar si tenemos las columnas mínimas
    if 'best_fitness' not in df.columns or 'avg_fitness' not in df.columns:
        print(f"Error: Missing fitness columns. Found: {df.columns.tolist()}")
        return

    # 3. DETERMINAR ESCALA Y LÍMITES (EVITA EL EFECTO ACHAPARRADO)
    max_f = df['best_fitness'].max()
    
    if max_f < 5:  # ESCALA NORMALIZADA (Vieja 0.0 - 2.0)
        hito_uno = 0.6
        hito_ambos = 2.0
        # Ajustamos el zoom para que el máximo no sea forzado a 2.0 si no hemos llegado
        upper_limit = max(max_f * 1.15, hito_uno * 1.2)
    else:          # ESCALA MASIVA (Nueva V29)
        hito_uno = 150000
        hito_ambos = 350000
        # Ajustamos el zoom para ver los hitos sociales
        upper_limit = max(max_f * 1.15, hito_ambos * 1.1)

    # 4. CONFIGURACIÓN DE LA FIGURA
    plt.rcParams['font.family'] = 'sans-serif' # Fuente con serifa, estilo paper
    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    # EJE IZQUIERDO: FITNESS
    ax1.set_xlabel('Generations', fontsize=12)
    ax1.set_ylabel('Cooperative Fitness Scale', color='tab:red', fontsize=12, fontweight='bold')
    
    # Graficar Fitness
    line1 = ax1.plot(df['gen'], df['best_fitness'], color='tab:red', 
                    label='Elite Fitness (Best)', linewidth=1.5, zorder=3)
    line2 = ax1.plot(df['gen'], df['avg_fitness'], color='tab:orange', 
                    alpha=0.5, linestyle='--', label='Population Average', zorder=2)
    
    ax1.set_ylim(0, upper_limit) # <--- Aquí estaba el error, ahora está definido
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, linestyle='--', alpha=0.15)

    # 5. EJE DERECHO: TIEMPO
    lns = line1 + line2
    if 'time' in df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Generation Time (sec)', color='tab:blue', fontsize=12)
        line3 = ax2.plot(df['gen'], df['time'], color='tab:blue', 
                        alpha=0.25, label='Computation Time', linewidth=1.5, zorder=1)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        lns = lns + line3
        
    labs = [l.get_label() for l in lns]

    # 6. LEYENDA (Esquina Inferior Derecha)
    ax1.legend(lns, labs, loc=leg_loc, frameon=True, 
               facecolor='white', fontsize=9, borderaxespad=1.5)

    # 7. TÍTULOS E HITOS (DINÁMICOS)
    plt.title("Evolutionary Progress Analysis: Social\nMulti-Agent Coordination (GRU Net)", 
              fontsize=14, pad=20)

    # Dibujar Hito 1 (Siempre visible)
    ax1.axhline(y=hito_uno, color='gray', linestyle=':', alpha=0.5)
    ax1.text(df['gen'].min(), hito_uno * 1.02, "Threshold: Single Agent Success", 
             color='gray', fontsize=9, fontstyle='italic')

    # Dibujar Hito de Cooperación (Solo si el zoom lo permite)
    if hito_ambos <= upper_limit:
        ax1.axhline(y=hito_ambos, color='green', linestyle=':', alpha=0.5)
        ax1.text(df['gen'].min(), hito_ambos * 1.01, "Cooperation Zone (Swarm Synergy)", 
                 color='green', fontsize=9, fontweight='bold')
    else:
        # Mensaje discreto si el objetivo está fuera de vista
        ax1.text(df['gen'].median(), upper_limit * 0.92, 
                 f"Target Cooperation ({hito_ambos}) is above current scale", 
                 color='green', fontsize=8, alpha=0.6, ha='center')

    fig.tight_layout()
    
    # Guardar imagen
    output_img = log_path.replace(".csv", "_academic_plot.png")
    #plt.savefig(output_img, dpi=300)
    
    print("-" * 40)
    print(f"EFSIM SYSTEM: Plotting Complete")
    print(f"Processed: {log_path}")
    print(f"Saved: {output_img}")
    print("-" * 40)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    # NUEVO: Argumento para la posición de la leyenda
    parser.add_argument("--loc", type=str, default="lower left", help="Legend location (e.g., upper left, lower right, best)")
    
    args = parser.parse_args()
    # Pasa el nuevo argumento a la función
    plot_evolution_results(args.log, args.loc)
