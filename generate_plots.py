import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 1. Silenciamos avisos innecesarios
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración estética
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.6)

def plot_results():
    def load_clean(name, label):
        # Carga robusta para tabuladores o comas
        df = pd.read_csv(name, sep=None, engine='python', skipinitialspace=True)
        df.columns = df.columns.str.strip()
        for col in ['goal_gap', 'success', 'step_count']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Condition'] = label 
        return df

    try:
        # Carga de datos
        s_df = load_clean('logs/social_results.csv', 'Social (Active)')
        m_df = load_clean('logs/mute_results.csv', 'Mute (Ablation)')
        n_df = load_clean('logs/noise_results.csv', 'Noise (Shuffled)')

        combined_df = pd.concat([s_df, m_df, n_df], ignore_index=True)
        gap_data = combined_df[combined_df['goal_gap'] >= 0].copy()
        
        # Definimos el orden y los colores para que sean consistentes en todos los gráficos
        order = ['Social (Active)', 'Mute (Ablation)', 'Noise (Shuffled)']
        my_palette = "Set2"

        # ---------------------------------------------------------
        # 1. GOAL GAP (Figura 7) - CORREGIDO SIN WARNINGS
        # ---------------------------------------------------------
        plt.figure(figsize=(12, 7))
        ax = sns.boxplot(
            data=gap_data, 
            x='Condition', 
            y='goal_gap', 
            order=order,
            hue='Condition',      # Fix para el warning
            palette=my_palette,
            legend=False,         # Fix para el warning
            showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"}
        )
        
        plt.title('Social Latency: Impact of Communication on Coordination', pad=25)
        plt.ylabel('Goal Gap (Time steps between arrivals)')
        plt.xlabel('Experimental Condition')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.savefig('goal_gap_comparison.png', dpi=300, bbox_inches='tight')
        print("[OK] goal_gap_comparison.png (Sin warnings)")

        # ---------------------------------------------------------
        # 2. SUCCESS RATE - COLORES ALINEADOS
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        rates = combined_df.groupby('Condition')['success'].mean() * 100
        rates = rates.reindex(order)

        # Usamos la misma paleta que el boxplot para coherencia visual
        colors = sns.color_palette(my_palette, n_colors=3)
        bars = plt.bar(rates.index, rates.values, color=colors, alpha=0.8, edgecolor='black')
        
        plt.ylim(0, 115)
        plt.ylabel('Group Success Percentage (%)')
        plt.title('Global Success Rate Comparison', pad=20)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', 
                     ha='center', fontweight='bold', fontsize=14)

        plt.savefig('success_rate.png', dpi=300, bbox_inches='tight')
        print("[OK] success_rate.png (Colores consistentes)")

        # ---------------------------------------------------------
        # 3. SIGNALING EFFICIENCY (Gráfico de dispersión)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        social_df = combined_df[combined_df['Condition'] == 'Social (Active)'].copy()
        # Procesar signals 'X|Y' -> X
        social_df['leader_signals'] = social_df['signal_counts'].apply(
            lambda x: float(str(x).split('|')[0]) if isinstance(x, str) and '|' in x else 0
        )
        valid_social = social_df[social_df['success'] == 1]

        sns.regplot(data=valid_social, x='leader_signals', y='step_count', 
                    scatter_kws={'alpha':0.4, 'color':'#66c2a5'}, 
                    line_kws={'color':'#e74c3c', 'lw':3})
        
        plt.title('Communication Efficiency: Signaling vs. Task Duration', pad=20)
        plt.xlabel('Leader Signals (Screams emitted)')
        plt.ylabel('Total Steps to Success')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig('signaling_efficiency.png', dpi=300, bbox_inches='tight')
        print("[OK] signaling_efficiency.png (Limpio)")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_results()
