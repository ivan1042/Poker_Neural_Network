import pandas as pd

existing = pd.read_csv("../parsed_save2.csv", index_col=0)

# 0: fold/ check, 1: call, 2: raise >1/2, 3: raise <1/2
def categorizer(row):
    if row['hero_bet_ratio'] <= 0:
        return 0
    elif row['facing_ratio'] == row['hero_bet_ratio']:
        return 1
    elif row['facing_ratio'] < row['hero_bet_ratio']:
       if row['hero_bet_ratio'] - row['facing_ratio'] <= 1/2:
            return 2
       else:
            return 3


existing['category'] = existing.apply(categorizer, axis=1)
existing = existing.fillna(1)
"""df = existing[(existing['category'] != 0) & (existing['category'] != 1)]
plt.plot(df.index, df['hero_bet_ratio'] - df['facing_ratio'])
plt.ylim(0, 2)
plt.show()"""
temp = existing['result']
existing = existing.drop(columns=['hero_bet_ratio', 'if_fold', 'result'])
existing['result'] = temp
existing.to_csv("simplified_data.csv")
print(existing.info())
