import pandas as pd

# Chargement des données
dev_df = pd.read_csv("stats_dev.csv")
std_df = pd.read_csv("stats_std.csv")
diff_df = pd.read_csv("stats_diff.csv")

alpha_cols = ['0', '1', '5', '10', '50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '800', '900', '1000', '2000']

# Calcul des métriques
results = []
for col in alpha_cols:
    results.append({
        "alpha": int(col),
        "dev": dev_df[col].mean(),
        "std": std_df[col].mean(),
        "diff": diff_df[col].mean()
    })

df = pd.DataFrame(results)

# Filtrage Pareto (min dev, min std, max diff)
def is_dominated(a, b):
    return (
        (a["dev"] >= b["dev"] and
         a["std"] >= b["std"] and
         a["diff"] <= b["diff"]) and
        (a["dev"] > b["dev"] or
         a["std"] > b["std"] or
         a["diff"] < b["diff"])
    )

pareto = []
for i, row in df.iterrows():
    dominated = False
    for j, other in df.iterrows():
        if i != j and is_dominated(row, other):
            dominated = True
            break
    if not dominated:
        pareto.append(row)

pareto_df = pd.DataFrame(pareto)
print(pareto_df.sort_values(by="alpha"))
