import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Chargement des données
dev_df = pd.read_csv("stats_dev.csv")
std_df = pd.read_csv("stats_std.csv")
diff_df = pd.read_csv("stats_diff.csv")

# Liste ordonnée des alphas
alpha_cols = ['0', '1', '5', '10', '50', '100', '150', '200', '250', '300',
              '400', '500', '600', '700', '800', '900', '1000', '2000']

# Calcul des moyennes et écart-types
data = []
for col in alpha_cols:
    data.append({
        "alpha": int(col),
        "dev_median": dev_df[col].median(),
        "std_median": std_df[col].median(),
        "diff_median": diff_df[col].median(),
        "dev_std": dev_df[col].std(),
        "std_std": std_df[col].std(),
        "diff_std": diff_df[col].std()
    })
df = pd.DataFrame(data).sort_values(by="alpha").reset_index(drop=True)

# Normalisation de toutes les colonnes numériques sauf "alpha"
scaler = MinMaxScaler()
columns_to_normalize = ["dev_median", "std_median", "diff_median", "dev_std", "std_std", "diff_std"]
df[[f"{col}_norm" for col in columns_to_normalize]] = scaler.fit_transform(df[columns_to_normalize])

# Extraction du front de Pareto (sur les moyennes uniquement pour l’instant)
def is_dominated(a, b):
    return (
        (a["dev_median"] >= b["dev_median"]) and
        (a["std_median"] >= b["std_median"]) and
        (a["diff_median"] <= b["diff_median"]) and
        ((a["dev_median"], a["std_median"], -a["diff_median"]) != (b["dev_median"], b["std_median"], -b["diff_median"]))
    )

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

# Balanced slope score (moyenne et écart-type normalisées)
epsilon = 1e-6
balanced_scores = []

for i in range(len(pareto_df)):
    if i == 0 or i == len(pareto_df) - 1:
        balanced_scores.append(float('-inf'))
        continue

    prev = pareto_df.loc[i - 1]
    next_ = pareto_df.loc[i + 1]

    # Deltas sur les moyennes
    delta_dev_mean = next_["dev_median_norm"] - prev["dev_median_norm"]
    delta_std_mean = next_["std_median_norm"] - prev["std_median_norm"]
    delta_diff_mean = next_["diff_median_norm"] - prev["diff_median_norm"]

    # Deltas sur les écarts-types
    delta_dev_std = next_["dev_std_norm"] - prev["dev_std_norm"]
    delta_std_std = next_["std_std_norm"] - prev["std_std_norm"]
    delta_diff_std = next_["diff_std_norm"] - prev["diff_std_norm"]

    # Gains utiles (baisse des moyennes et écarts-types, hausse de diff)
    useful_gains = 0
    for d in [-delta_dev_mean, -delta_std_mean, delta_diff_mean, -delta_dev_std, -delta_std_std, -delta_diff_std]:
        if d > 0:
            useful_gains += d

    # Sacrifices subis
    sacrifices = 0
    for d in [delta_dev_mean, delta_std_mean, -delta_diff_mean, delta_dev_std, delta_std_std, delta_diff_std]:
        if d > 0:
            sacrifices += d

    score = useful_gains / (sacrifices + epsilon)
    balanced_scores.append(score)

# Ajout et tri
pareto_df["balanced_score"] = balanced_scores
pareto_df = pareto_df.sort_values(by="balanced_score", ascending=False)

# Affichage
print("Solutions optimales avec prise en compte de la stabilité intra-sujet :")
print(pareto_df[["alpha", "dev_median", "std_median", "diff_median", "dev_std", "std_std", "diff_std", "balanced_score"]])
