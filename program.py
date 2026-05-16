import pandas as pd

df = pd.read_csv("data_penyakit.csv")
df.info()

df = df.dropna()
print(f"{df.isnull().sum()}")

df.to_csv("data_bersih.csv", index=False)