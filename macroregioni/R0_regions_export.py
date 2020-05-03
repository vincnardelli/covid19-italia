# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# The SIR model differential equations.
def deriv(y, t, N, beta,gamma):
    S,I,R = y

    dSdt = -(beta*I/N)*S 
    dIdt = (beta*S/N)*I - gamma*I 
    dRdt = gamma*I 
    
    return dSdt, dIdt, dRdt

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

vector_regions = ['nord', 'centro', 'sud', 'isole']#,'nolombardia','lombardia']
time_window    = 5
for r in range(len(vector_regions)):
    fit_region = vector_regions[r]

    if fit_region =='nord':
        region    = ['Lombardia','Veneto','Emilia-Romagna','Liguria','Piemonte','Valle d\'Aosta','P.A. Trento','P.A. Bolzano','Friuli Venezia Giulia'] 
        n_regions = len(region)
    elif fit_region =='centro':
        region    = ['Toscana','Marche','Umbria','Lazio','Abruzzo','Molise']
        n_regions = len(region)
    elif fit_region =='sud':
        region    = ['Puglia','Calabria','Basilicata','Campania']
        n_regions = len(region)
    elif fit_region =='isole':
        region    = ['Sicilia','Sardegna']
        n_regions = len(region)
    
    elif  fit_region =='italia': 
        region    = 'Italia'
        n_regions = 1
    elif fit_region =='nolombardia':
        region    = ['Abruzzo','Basilicata','P.A. Bolzano','Calabria','Campania','Emilia-Romagna','Friuli Venezia Giulia','Lazio','Liguria','Marche','Molise','Piemonte','Puglia','Sardegna','Sicilia','Toscana','P.A. Trento','Umbria','Valle d\'Aosta','Veneto']
        n_regions = len(region)    
    elif fit_region =='lombardia':
        region    = ['Lombardia']
        n_regions = 1
    print(fit_region)

    popolation_regions = np.array([  1304970,      559084,        533050,   1947131,   5801692,         4459477,                1215220,5879082, 1550640,    10060574,  1525271,  305617,    4356406, 4029053, 1639591,  4999891,  3729641,       541380,  882015,          125666, 4905854])
    name_regions       = np.array(['Abruzzo','Basilicata','P.A. Bolzano','Calabria','Campania','Emilia-Romagna','Friuli Venezia Giulia','Lazio','Liguria','Lombardia','Marche','Molise','Piemonte','Puglia','Sardegna','Sicilia','Toscana','P.A. Trento','Umbria','Valle d\'Aosta','Veneto'])
    regions            = np.vstack((name_regions,popolation_regions))

    mask_reg = []
    for i in range(n_regions):
        mask_reg.append(regions[0,:] == region[i])
    mask_reg = np.array(mask_reg)
    
    data = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    N = 0
    xxx = []
    yyy = []
    zzz = []
    for i in range(n_regions):
        N += int(regions[1,mask_reg[i]])
        mask_REG=data['denominazione_regione']==region[i]
        xxx.append(data.loc[mask_REG,'totale_casi'])
        yyy.append(data.loc[mask_REG,'deceduti'])
        zzz.append(data.loc[mask_REG,'dimessi_guariti'])

    ydata       = np.array(np.sum(xxx,axis=0))
    ydata_death = np.array(np.sum(yyy,axis=0))
    ydata_rec   = np.array(np.sum(zzz,axis=0))
    ydata_inf   = ydata-ydata_rec-ydata_death
    xdata       = pd.to_numeric(range(ydata.shape[0]))
    today       = len(xdata)
    
    def minimizer(R0,t1=today-time_window,t2=today): #7
    
        #true data
        ydata_inf_2=np.array(ydata_inf[t1:t2])
        xdata_2=np.arange(0,len(ydata_inf_2))

        #model
        fin_result=time_evo(N,0.07*R0,0.07,I0=ydata_inf_2[0])
        i_vec=fin_result[2]
        i_vec_2=i_vec[0:len(xdata_2)]

        #average error
        error=np.sum(np.abs(ydata_inf_2-i_vec_2)/ydata_inf_2)*100

        return error

    minimizer_vec=np.vectorize(minimizer)

    xgrid    = np.arange(0.1,1.3,0.01)
    ygrid    = minimizer_vec(xgrid)
    r0_ideal = round(xgrid[np.argmin(ygrid)],2)
    print('r0_ideal for the '+fit_region+': ',r0_ideal)

    ydata_inf_2 = np.array(ydata_inf[today-time_window:today])
    xdata_2     = np.arange(0,len(ydata_inf_2))
    print('ydata_inf.shape '+fit_region+': ',ydata_inf.shape)
    print('ydata_inf for the '+fit_region+': ',ydata_inf)
    print('ydata_inf_2 for the '+fit_region+': ',ydata_inf_2)

    fin_result  = time_evo(N,0.07*r0_ideal,0.07,I0=ydata_inf_2[0])

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
    
    for i in range(today-(time_window-1)): 
        min_val=minimizer_gen(i,i+time_window) 
        r0_time.append(min_val)
        print(i,min_val)
      
    if fit_region =='nord':
        r0_time_nord=np.array(r0_time)
    elif fit_region =='centro':
        r0_time_centro=np.array(r0_time)
    elif fit_region =='sud':
        r0_time_sud=np.array(r0_time)
    elif fit_region =='isole':
        r0_time_isole=np.array(r0_time)
    elif fit_region =='nolombardia':
        r0_time_nolombardia=np.array(r0_time)
    elif fit_region =='lombardia':
        r0_time_lombardia=np.array(r0_time)
    r0_time.clear()
    
df_r0=pd.DataFrame(pd.to_datetime(np.arange(len(r0_time_nord)),unit='D',origin='2020-02-28'))
df_r0['nord']   = r0_time_nord
df_r0['centro'] = r0_time_centro
df_r0['sud']    = r0_time_sud
df_r0['isole']  = r0_time_isole
#df_r0['nolombardia']    = r0_time_nolombardia
#df_r0['lombardia']      = r0_time_lombardia


df_r0.columns   = ['Data','nord','centro','sud','isole']#,'nolombardia','lombardia']

df_r0.to_csv('output/r0_regions.csv',index=False)


# %%


