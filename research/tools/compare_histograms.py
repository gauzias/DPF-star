import numpy as np
import scipy.stats as stats
import itertools
import pandas as pd

def chi_square_test(hist1, hist2):
    """
    Test du Chi-2 pour comparer deux histogrammes en normalisant leurs sommes.
    """
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    chi2, p = stats.chisquare(hist1, hist2)
    return chi2, p

def kolmogorov_smirnov_test(hist1, hist2):
    """
    Test de Kolmogorov-Smirnov pour comparer deux distributions.
    """
    ks_stat, p_value = stats.ks_2samp(hist1, hist2)
    return ks_stat, p_value

def bhattacharyya_distance(hist1, hist2):
    """
    Distance de Bhattacharyya pour mesurer la similarité entre deux histogrammes.
    """
    return -np.log(np.sum(np.sqrt(hist1 * hist2)))

def kullback_leibler_divergence(hist1, hist2):
    """
    Divergence de Kullback-Leibler pour mesurer la différence d'information.
    """
    hist1 = np.where(hist1 == 0, 1e-10, hist1)  # Évite les divisions par zéro
    hist2 = np.where(hist2 == 0, 1e-10, hist2)
    return np.sum(hist1 * np.log(hist1 / hist2))

def interpret_results(df_results):
    """
    Interprète les résultats et donne une conclusion sur la similarité des histogrammes.
    """
    threshold_p_value = 0.05
    significant_differences = 0
    total_comparisons = len(df_results)
    
    print("\n### Analyse des résultats ###\n")
    
    for _, row in df_results.iterrows():
        print(f"Comparaison entre Histogramme {row['Histogramme 1']} et Histogramme {row['Histogramme 2']}:")
        print(f"  - Test du Chi-2 : Stat={row['Chi-2']:.4f}, p={row['p-valeur Chi-2']:.4f}")
        print(f"  - Test KS : Stat={row['Kolmogorov-Smirnov']:.4f}, p={row['p-valeur KS']:.4f}")
        print(f"  - Distance de Bhattacharyya : {row['Distance Bhattacharyya']:.4f}")
        print(f"  - Divergence KL : {row['Divergence KL']:.4f}")
        
        if row['p-valeur Chi-2'] < threshold_p_value or row['p-valeur KS'] < threshold_p_value:
            print("  => Les histogrammes sont significativement différents.\n")
            significant_differences += 1
        else:
            print("  => Les histogrammes ne présentent pas de différences significatives.\n")
    
    if significant_differences == 0:
        print("Conclusion : Aucun histogramme ne présente de différence significative, ils semblent similaires.\n")
    elif significant_differences == total_comparisons:
        print("Conclusion : Toutes les paires d'histogrammes sont significativement différentes, ils représentent des distributions distinctes.\n")
    else:
        print("Conclusion : Certaines paires d'histogrammes sont similaires, tandis que d'autres montrent des différences significatives.\n")

def compare_histograms(histograms):
    """
    Compare tous les histogrammes donnés en entrée deux par deux.
    """
    results = []
    
    for (i, hist1), (j, hist2) in itertools.combinations(enumerate(histograms), 2):
        chi2, p_chi2 = chi_square_test(hist1, hist2)
        ks_stat, p_ks = kolmogorov_smirnov_test(hist1, hist2)
        bhattacharyya = bhattacharyya_distance(hist1, hist2)
        kl_div = kullback_leibler_divergence(hist1, hist2)
        
        results.append({
            "Histogramme 1": i,
            "Histogramme 2": j,
            "Chi-2": chi2,
            "p-valeur Chi-2": p_chi2,
            "Kolmogorov-Smirnov": ks_stat,
            "p-valeur KS": p_ks,
            "Distance Bhattacharyya": bhattacharyya,
            "Divergence KL": kl_div
        })
    
    df_results = pd.DataFrame(results)
    return df_results

# Exemple d'utilisation
def generate_example_histograms():
    """Génère des histogrammes factices pour test"""
    np.random.seed(42)
    hist1, _ = np.histogram(np.random.normal(0, 1, 1000), bins=10, density=False)
    hist2, _ = np.histogram(np.random.normal(0.5, 1, 1000), bins=10, density=False)
    hist3, _ = np.histogram(np.random.normal(0, 1.5, 1000), bins=10, density=False)
    return [hist1, hist2, hist3]

# Génération et comparaison
test_histograms = generate_example_histograms()
result_table = compare_histograms(test_histograms)

# Affichage des résultats
import ace_tools_open as atools
atools.display_dataframe_to_user(name="Résultats Comparaison Histogrammes", dataframe=result_table)

# Interprétation et conclusion
interpret_results(result_table)