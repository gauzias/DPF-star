import pandas as pd

# Chargement des données
dev_df = pd.read_csv("stats_dev.csv")
std_df = pd.read_csv("stats_std.csv")
diff_df = pd.read_csv("stats_diff.csv")

# Liste ordonnée des alphas
alpha_cols = ['0', '1', '5', '10', '50', '100', '150', '200', '250', '300',
              '400', '500', '600', '700', '800', '900', '1000', '2000']

# Calcul des moyennes
data = []
for col in alpha_cols:
    data.append({
        "alpha": int(col),
        "dev": dev_df[col].mean(),
        "std": std_df[col].mean(),
        "diff": diff_df[col].mean()
    })
df = pd.DataFrame(data).sort_values(by="alpha").reset_index(drop=True)

# Fonction de dominance (Pareto pur)
def is_dominated(a, b):
    return (
        (a["dev"] >= b["dev"]) and
        (a["std"] >= b["std"]) and
        (a["diff"] <= b["diff"]) and
        ((a["dev"], a["std"], -a["diff"]) != (b["dev"], b["std"], -b["diff"]))
    )

# Extraire le front de Pareto
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

# Calcul de la pente locale (slope-based filter)
slope_scores = []
for i in range(len(pareto_df)):
    if i == 0 or i == len(pareto_df) - 1:
        slope_scores.append(float('-inf'))  # Bords = score minimal
        continue

    prev = pareto_df.loc[i - 1]
    curr = pareto_df.loc[i]
    next_ = pareto_df.loc[i + 1]

    delta_diff = next_["diff"] - prev["diff"]
    delta_dev = next_["dev"] - prev["dev"]
    delta_std = next_["std"] - prev["std"]

    # On veut un bon gain en diff pour un faible coût en dev et std
    cost = abs(delta_dev) + abs(delta_std)
    slope = delta_diff / cost if cost != 0 else 0
    slope_scores.append(slope)

# Ajout à la DataFrame
pareto_df["slope_score"] = slope_scores

# Trier par score décroissant (meilleurs compromis d'abord)
pareto_df = pareto_df.sort_values(by="slope_score", ascending=False)

# Affichage
print("Solutions Pareto triées par efficacité de compromis (slope-based):")
print(pareto_df[["alpha", "dev", "std", "diff", "slope_score"]])
