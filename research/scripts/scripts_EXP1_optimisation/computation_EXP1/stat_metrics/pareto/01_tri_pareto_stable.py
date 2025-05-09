import pandas as pd

# Chargement des fichiers
dev_df = pd.read_csv("stats_dev.csv")
std_df = pd.read_csv("stats_std.csv")
diff_df = pd.read_csv("stats_diff.csv")

# Liste ordonnée des alphas
alpha_cols = ['0', '1', '5', '10', '50', '100', '150', '200', '250', '300',
              '400', '500', '600', '700', '800', '900', '1000', '2000']

# Calcul des moyennes pour chaque alpha
data = []
for col in alpha_cols:
    dev = dev_df[col].mean()
    std = std_df[col].mean()
    diff = diff_df[col].mean()
    data.append({"alpha": int(col), "dev": dev, "std": std, "diff": diff})

df = pd.DataFrame(data)

# Fonction de dominance (Pareto)
def is_dominated(a, b):
    return (
        (a["dev"] >= b["dev"]) and
        (a["std"] >= b["std"]) and
        (a["diff"] <= b["diff"]) and
        ((a["dev"], a["std"], -a["diff"]) != (b["dev"], b["std"], -b["diff"]))
    )

# Détection du front de Pareto
pareto_front = []
for i, a in df.iterrows():
    dominated = False
    for j, b in df.iterrows():
        if is_dominated(a, b):
            dominated = True
            break
    if not dominated:
        pareto_front.append(a)

pareto_df = pd.DataFrame(pareto_front).sort_values(by="alpha").reset_index(drop=True)

# Ajout du score de stabilité (variation locale)
stability_scores = []
for i in range(len(pareto_df)):
    if i == 0 or i == len(pareto_df) - 1:
        # Bords : pas de voisin gauche/droite, on fixe un score élevé
        stability_scores.append(float('inf'))
        continue

    prev = pareto_df.loc[i - 1]
    curr = pareto_df.loc[i]
    next_ = pareto_df.loc[i + 1]

    delta_dev = abs(next_["dev"] - prev["dev"]) / 2
    delta_std = abs(next_["std"] - prev["std"]) / 2
    delta_diff = abs(next_["diff"] - prev["diff"]) / 2

    # Score basé sur somme des deltas (moins c’est stable, plus c’est élevé)
    instability = delta_dev + delta_std - delta_diff
    stability_scores.append(instability)

# Ajout à la DataFrame
pareto_df["stability_score"] = stability_scores

# Tri des solutions stables
pareto_df = pareto_df.sort_values(by="stability_score")

# Affichage
print("Solutions Pareto stables (triées par stabilité croissante):")
print(pareto_df[["alpha", "dev", "std", "diff", "stability_score"]])

