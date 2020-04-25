# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import geopandas as gpd

# %% [markdown]
# ### Definition of the model

# %%
# The SIR model differential equations.
def deriv(y, t, N, beta,gamma):
    S,I,R = y

    dSdt = -(beta*I/N)*S 
    dIdt = (beta*S/N)*I - gamma*I 
    dRdt = gamma*I 
    
    return dSdt, dIdt, dRdt

# %% [markdown]
# ### Integration of the differential equations

# %%
def time_evo(N,beta,gamma,I0=1,R0=0,t=np.arange(0,365)):
    # Definition of the initial conditions
    # I0 and R0 denotes the number of initial infected people (I0) 
    # and the number of people that recovered and are immunized (R0)
    
    # t ise the timegrid
    
    S0=N-I0-R0  # number of people that can still contract the virus
    
    # Initial conditions vector
    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N,beta,gamma))
    S, I, R = np.transpose(ret)
    
    return (t,S,I,R)

# %% [markdown]
# # All-in-one

# %%
popolation_regions = np.array([  1304970,      559084,        533050,   1947131,   5801692,         4459477,                1215220,5879082, 1550640,    10060574,  1525271,  305617,    4356406, 4029053, 1639591,  4999891,  3729641,       541380,  882015,          125666, 4905854])
name_regions       = np.array(['Abruzzo','Basilicata','P.A. Bolzano','Calabria','Campania','Emilia-Romagna','Friuli Venezia Giulia','Lazio','Liguria','Lombardia','Marche','Molise','Piemonte','Puglia','Sardegna','Sicilia','Toscana','P.A. Trento','Umbria','Valle d\'Aosta','Veneto'])
     
data = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')

df_r0=pd.DataFrame(data['data'].tail(1))

for region in name_regions:
    N = popolation_regions[name_regions == region]
    ydata       = np.array(data.loc[data['denominazione_regione'] == region, "totale_casi"])
    ydata_death = np.array(data.loc[data['denominazione_regione'] == region, "deceduti"])
    ydata_rec   = np.array(data.loc[data['denominazione_regione'] == region, "dimessi_guariti"])
    ydata_inf   = ydata-ydata_rec-ydata_death
    xdata       = pd.to_numeric(range(ydata.shape[0]))
    today       = len(xdata)

    def minimizer(R0,t1=today-5,t2=today):
    
        #true data
        ydata_inf_2=np.array(ydata_inf[t1:t2])
        xdata_2=np.arange(0,len(ydata_inf_2))

        #model
        fin_result=time_evo(N,0.1*R0,0.1,I0=ydata_inf_2[0])
        i_vec=fin_result[2]
        i_vec_2=i_vec[0:len(xdata_2)]

        #average error
        error=np.sum(np.abs(ydata_inf_2-i_vec_2)/ydata_inf_2)*100

        return error

    minimizer_vec=np.vectorize(minimizer)

    xgrid    = np.arange(1,1.3,0.01)
    ygrid    = minimizer_vec(xgrid)
    r0_ideal = round(xgrid[np.argmin(ygrid)],2)
    print('r0_ideal for the '+region+': ',r0_ideal)

    ydata_inf_2 = np.array(ydata_inf[today-5:today])
    xdata_2     = np.arange(0,len(ydata_inf_2))
    print('ydata_inf.shape '+region+': ',ydata_inf.shape)
    print('ydata_inf for the '+region+': ',ydata_inf)
    print('ydata_inf_2 for the '+region+': ',ydata_inf_2)

    fin_result  = time_evo(N,0.1*r0_ideal,0.1,I0=ydata_inf_2[0])

    t=fin_result[0]
    s_vec=fin_result[1]
    i_vec=fin_result[2]
    r_vec=fin_result[3]
    
    
    def minimizer_gen(t1,t2):

        xgrid=np.arange(0.1,7.2,0.01)
        ygrid=minimizer_vec(xgrid,t1=t1,t2=t2)
        r0_ideal=round(xgrid[np.argmin(ygrid)],2)

        return r0_ideal  
    
    r0_time=[]
    
  #  for i in range(today-4):
  #      min_val=minimizer_gen(i,i+5)
   #     r0_time.append(min_val)
   #     print(i,min_val)

    min_val=minimizer_gen(today-7,today)
 

    df_r0[region]   = min_val
    r0_time.clear()


# %%
df = df_r0.T
df['description'] = df.index
df.rename(columns={ df.columns[0]: "R0" }, inplace = True)
df = df.iloc[1:]
df_row = pd.DataFrame([{"description": "Trentino", "R0":1}])
df = pd.concat([df, df_row], ignore_index=True)
df['description'][df.description == "Friuli Venezia Giulia"] = "Friuli-Venezia Giulia"
trentino = round(float((sum(df.R0[df.description == "P.A. Trento"], df.R0[df.description == "P.A. Bolzano"])/2)), 2)
row_df = pd.DataFrame([{'R0':trentino, "description":"Trentino-Alto Adige"}])
df = pd.concat([df, row_df], ignore_index=True)


# %%
map = gpd.read_file("regioni_italiane.geojson")
map = map.merge(df, on='description', how='left')
map.to_file("export/r0_regioni.geojson", driver='GeoJSON')
classificazione = pd.read_excel('classificazione_regioni.xlsx')
map = map.merge(classificazione, on='description', how='left')
map[["description", "R0", "Area"]].to_csv("export/r0_regioni.csv")


# %%


