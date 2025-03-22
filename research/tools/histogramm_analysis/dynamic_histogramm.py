import matplotlib.pyplot as plt
import math

def plot_histograms(*lists, bins="auto"):
    n = len(lists)
    cols = min(5, n)  # Maximum de 5 subplots par ligne
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]  # Assurer une itération correcte
    
    for i, data in enumerate(lists):
        axes[i].hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_title(f'Histogram {i+1}')
    
    # Cacher les subplots inutilisés
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
#plot_histograms([1, 2, 2, 3, 3, 3, 4, 4, 5], [10, 20, 20, 30, 30, 30, 40], [5, 15, 25, 35, 45, 55], [2, 3, 5, 7, 11], [1, 1, 2, 3, 5, 8], [0, 10, 20, 30])
