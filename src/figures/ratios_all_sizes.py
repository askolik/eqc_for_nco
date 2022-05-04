
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

data = {
    'approximation_ratio': [],
    'algorithm': [],
    'tsp_type': []
}

for n_vars in [5, 10, 20]:
    for algo_type in ['EQC', 'NEQC', 'NN']:  # , 'HWETE', 'HWE']:
        try:
            with open(f'../../data/validation_performance/tsp{n_vars}_{algo_type.lower()}_validation100_10agents.pickle', 'rb') as file:
                ar_vals = pickle.load(file)
                for val in ar_vals:
                    data['approximation_ratio'].append(val)
                    data['algorithm'].append(algo_type)
                    data['tsp_type'].append(f'TSP{n_vars}')
        except:
            print(f"Error at n_vars {n_vars}, algo_type {algo_type}")

df = pd.DataFrame(data)

print(df)
# exit()

ax = sns.boxplot(x="tsp_type", y="approximation_ratio",
            hue="algorithm", # palette=["m", "g"],
            data=df)

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['EQC', 'NEQC', 'NN'])

plt.ylabel("Approximation ratio")
plt.xlabel("")

plt.axhline(y=1.5, color='black', linestyle='dotted', linewidth=1)

plt.show()