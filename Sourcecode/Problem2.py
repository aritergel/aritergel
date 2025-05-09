import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load player statistics from CSV
df = pd.read_csv("results.csv")

# Keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Convert non-numeric (like "N/a") to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === I. Top 3 highest and lowest players per statistic ===
top3_text = ""
for col in numeric_cols:
    top3_high = df.sort_values(by=col, ascending=False)[["Name", col]].head(3)
    top3_low = df.sort_values(by=col, ascending=True)[["Name", col]].head(3)
    top3_text += f"\nTop 3 Highest for {col}:\n{top3_high.to_string(index=False)}\n"
    top3_text += f"Top 3 Lowest for {col}:\n{top3_low.to_string(index=False)}\n"

with open("top_3.txt", "w") as f:
    f.write(top3_text)

print("File saved: top_3.txt")

# === II. Summary: median, mean, std for each stat ===
summary = {
    "Median": df[numeric_cols].median(),
    "Mean": df[numeric_cols].mean(),
    "Std": df[numeric_cols].std()
}
summary_df = pd.DataFrame(summary).T
summary_df["Team"] = "All"

# Summary by team
if 'stats_standard_Team' in df.columns:
    team_results = []
    for team in df['stats_standard_Team'].dropna().unique():
        team_df = df[df['stats_standard_Team'] == team]
        med = team_df[numeric_cols].median()
        mean = team_df[numeric_cols].mean()
        std = team_df[numeric_cols].std()
        temp = pd.DataFrame({"Median": med, "Mean": mean, "Std": std})
        temp["Team"] = team
        team_results.append(temp.reset_index())

    full_stats = pd.concat([summary_df.reset_index()] + team_results, ignore_index=True)
    full_stats.rename(columns={"index": "Attribute"}, inplace=True)
    full_stats.to_csv("results2.csv", index=False)
    print("File saved: results2.csv")
else:
    print("⚠️ Team column not found in dataset!")

# === III. Histogram plot for each numeric attribute ===
os.makedirs("histograms", exist_ok=True)
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"histograms/{col}.png")
    plt.close()

print(" Histograms saved in the 'histograms' folder.")

# === IV. Identify best performing teams ===
if 'stats_standard_Team' in df.columns:
    team_avg = df.groupby('stats_standard_Team')[numeric_cols].mean()
    top_teams = team_avg.idxmax()
    print("\n Top Performing Teams by Statistic:")
    print(top_teams)
else:
    print(" Cannot identify best teams — 'Team' column missing.")

