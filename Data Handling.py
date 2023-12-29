import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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

#print(df_MS_num.corr())
#print(df_C_num.corr())


df_C_independent = df_C[["Length (ft)", "Year"]]

#Model for C, degree = 1
def model_C_1():
    lm1 = LinearRegression()
    lm1.fit(df_C_independent, df_C["Listing Price (USD)"])
    print(lm1.intercept_)
    print(lm1.coef_)

    p_predicted = lm1.predict(df_C_independent)

    #Visual Evaluation
    plt.figure(figsize=(10, 8))
    ax1 = sns.histplot(df_C['Listing Price (USD)'], color="r", label="Actual Value")
    sns.histplot(p_predicted, color="b", label="Fitted Values" , ax=ax1)
    plt.title('Actual vs Fitted Values for Price')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion')
    plt.show()
    plt.close()

    #Statistical Evaluation
    c_1_R2 = lm1.score(df_C_independent, df_C["Listing Price (USD)"])
    c_1_MSE = mean_squared_error(df_C['Listing Price (USD)'], p_predicted)
    print(c_1_R2)
    print(c_1_MSE)


#Model for C, degree > 1
def model_C(d):
    input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree = d, include_bias=False)), ('model',LinearRegression())]
    pipe=Pipeline(input)
    
    df_C_independent_model = df_C_independent.astype(float)
    pipe.fit(df_C_independent_model,df_C["Listing Price (USD)"])
    
    p_pipe=pipe.predict(df_C_independent_model)
    
    r2 = r2_score(df_C["Listing Price (USD)"], p_pipe)
    mse = mean_squared_error(df_C["Listing Price (USD)"], p_pipe)
    
    logreg_model = pipe.named_steps['model']

    #visual evaluation
    plt.figure(figsize=(10, 8))
    ax1 = sns.histplot(df_C['Listing Price (USD)'], color="r", label="Actual Value")
    sns.histplot(p_pipe, color="b", label="Fitted Values" , ax=ax1)
    plt.title('Actual vs Fitted Values for Price')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion')
    plt.show()
    plt.close()
    
    
    coefficients = logreg_model.coef_
    intercept = logreg_model.intercept_
    
    return [coefficients,intercept,r2,mse]

print(model_C(3))
    


