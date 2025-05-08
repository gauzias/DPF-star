import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

# Normalisation des colonnes (min-max)
scaler = MinMaxScaler()
normalized = scaler.fit_transform(df[["dev", "std", "diff"]])
df[["dev_norm", "std_norm", "diff_norm"]] = normalized

# Définition du test de dominance (sur valeurs originales, mais possible aussi sur normalisées)
def is_dominated(a, b):
    return (
        (a["dev"] >= b["dev"]) and
        (a["std"] >= b["std"]) and
        (a["diff"] <= b["diff"]) and
        ((a["dev"], a["std"], -a["diff"]) != (b["dev"], b["std"], -b["diff"]))
    )

# Extraction du front de Pareto
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

# Calcul du score de compromis équilibré sur les colonnes normalisées
balanced_scores = []
epsilon = 1e-6

for i in range(len(pareto_df)):
    if i == 0 or i == len(pareto_df) - 1:
        balanced_scores.append(float('-inf'))
        continue

    prev = pareto_df.loc[i - 1]
    next_ = pareto_df.loc[i + 1]

    delta_dev = next_["dev_norm"] - prev["dev_norm"]
    delta_std = next_["std_norm"] - prev["std_norm"]
    delta_diff = next_["diff_norm"] - prev["diff_norm"]

    useful_gains = 0
    if delta_diff > 0:
        useful_gains += delta_diff
    if delta_dev < 0:
        useful_gains += -delta_dev
    if delta_std < 0:
        useful_gains += -delta_std

    sacrifices = 0
    if delta_diff < 0:
        sacrifices += -delta_diff
    if delta_dev > 0:
        sacrifices += delta_dev
    if delta_std > 0:
        sacrifices += delta_std

    score = useful_gains / (sacrifices + epsilon)
    balanced_scores.append(score)

# Ajout et tri
pareto_df["balanced_score"] = balanced_scores
pareto_df = pareto_df.sort_values(by="balanced_score", ascending=False)

# Affichage final
print("Solutions Pareto triées (normalisées) par compromis multi-objectif :")
print(pareto_df[["alpha", "dev", "std", "diff", "balanced_score"]])
