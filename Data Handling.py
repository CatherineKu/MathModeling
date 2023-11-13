import pandas as pd
import numpy as np
import seaborn as sns

# Data import and cleaning
file_path_MS = 'Boats_MS.xlsx'
file_path_C = 'Boats_C.xlsx'

df_MS = pd.read_excel(file_path_MS)
df_C = pd.read_excel(file_path_C)

df_MS["Length (ft)"] = df_MS["Length (ft)"].astype(float)
df_MS = df_MS.dropna(subset = ["Country/Region/State"])

df_C["Year"] = df_MS["Length (ft)"].astype(int)

#only for testing purpose
#df_MS.info()
#df_C.info()




# Correlation between variables
df_MS_num = df_MS.select_dtypes(include=['number'])
df_C_num = df_C.select_dtypes(include=['number'])

print(df_MS_num.corr())
print(df_C_num.corr())

