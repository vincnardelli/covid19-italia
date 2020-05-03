# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# The SIR model differential equations.
def deriv(y, t, N, gamma,beta1,beta2,t_tresh=22):
    S,I,R = y

    if t<=t_tresh:
        B=beta1
    elif t>t_tresh and t<=1000:
        B=beta1*np.exp(-(t-t_tresh)/beta2)
    elif t>1000:
        B=0.2*np.exp(-(t-1000)/beta2)
    
    dSdt = -(B*I/N)*S 
    dIdt = (B*S/N)*I - gamma*I 
    dRdt = gamma*I 
    
    return dSdt, dIdt, dRdt

def time_evo(N,beta1,beta2,gamma,death_rate,t_tresh=22,I0=1,R0=0,t=np.arange(0,365)):
    # Definition of the initial conditions
    # I0 and R0 denotes the number of initial infected people (I0) 
    # and the number of people that recovered and are immunized (R0)
    
    # t ise the timegrid
    
    S0=N-I0-R0  # number of people that can still contract the virus
    
    # Initial conditions vector
    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N,gamma,beta1,beta2,t_tresh))
    S, I, R = np.transpose(ret)
    
    return (t,S,I,(1-death_rate/100)*R,R*death_rate/100)

vector_regions = ['nord', 'centro', 'sud', 'isole']#,'italia','nolombardia','lombardia']
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

    popolation_regions = np.array([  1304970,      559084,        533050,   1947131,   5801692,         4459477,                1215220,5879082, 1550640,    10060574,  1525271,  305617,    4356406, 4029053, 1639591,  4999891,  3729641,       541380,  882015,          125666, 4905854])
    name_regions       = np.array(['Abruzzo','Basilicata','P.A. Bolzano','Calabria','Campania','Emilia-Romagna','Friuli Venezia Giulia','Lazio','Liguria','Lombardia','Marche','Molise','Piemonte','Puglia','Sardegna','Sicilia','Toscana','P.A. Trento','Umbria','Valle d\'Aosta','Veneto'])
    regions            = np.vstack((name_regions,popolation_regions))

    mask_reg = []
    for i in range(n_regions):
        mask_reg.append(regions[0,:] == region[i])
    mask_reg = np.array(mask_reg)

    if region=='Italia':
        data = pd.read_csv('https://github.com/pcm-dpc/COVID-19/raw/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
        xdata=pd.to_numeric(range(data.shape[0]))
        ydata=data['totale_casi']
        ydata_death=data['deceduti']
        ydata_rec=data['dimessi_guariti']
        N = 60.48*10**6
        
    else:
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
        xdata       = pd.to_numeric(range(ydata.shape[0]))
        
    if fit_region =='nord':
        fin_result=time_evo(N,0.41,27.65,1/14,5.5,t_tresh=17,I0=2,t=np.arange(0,720)) # Nord + 0 giorni
        dt = 0
    elif fit_region =='centro':
        fin_result=time_evo(N,0.41,24.65,1/14,3.4,t_tresh=14.4,I0=2,t=np.arange(0,720)) # Centro + 12 giorni
        dt = 10
    elif fit_region =='sud':
        fin_result=time_evo(N,0.41,29.14,1/14,2.5,t_tresh=9,I0=2,t=np.arange(0,720))  # Sud + 12 giorni
        dt = 12
    elif fit_region =='isole':
        fin_result=time_evo(N,0.41,27.25,1/14,2,t_tresh=7.8,I0=2,t=np.arange(0,720)) # Isole + 16 giorni
        dt = 16      

    elif  fit_region =='italia': 
        fin_result=time_evo(N,0.415,28,1/14,6.5,t_tresh=17,I0=2,t=np.arange(0,720)) # Italia
        dt = 0
    if fit_region =='nolombardia':
        fin_result=time_evo(N,0.415,26.5,1/14,4.2,t_tresh=17,I0=2,t=np.arange(0,720)) # Nord + 0 giorni
        dt = 4
    if fit_region =='lombardia':
        fin_result=time_evo(N,0.415,25.85,1/14,8,t_tresh=17,I0=1,t=np.arange(0,720)) # Nord + 0 giorni
        dt = 0

    t=fin_result[0]
    s_vec=fin_result[1]
    i_vec=fin_result[2]
    r_vec=fin_result[3]
    m_vec=fin_result[4]

    ydata_inf=ydata-ydata_rec-ydata_death

    # Starting time for the model according to each region
    if fit_region   ==   'nord':
        new_t  = pd.to_datetime(t,unit='D',origin='2020-02-07') 
    elif fit_region == 'centro':
        new_t  = pd.to_datetime(t,unit='D',origin='2020-02-17')
    elif fit_region ==    'sud':
        new_t  = pd.to_datetime(t,unit='D',origin='2020-02-19') 
    elif fit_region ==  'isole':
        new_t  = pd.to_datetime(t,unit='D',origin='2020-02-23')
    elif fit_region == 'italia': 
        new_t  = pd.to_datetime(t,unit='D',origin='2020-02-07')
    elif fit_region == 'nolombardia': 
        new_t  = pd.to_datetime(t,unit='D',origin='2020-02-11')
    elif fit_region == 'lombardia': 
        new_t  = pd.to_datetime(t,unit='D',origin='2020-02-07')


    # Starting time for the data - All regions
    data_t        = pd.to_datetime(xdata,unit='D',origin='2020-02-24') 

    # Model dataframe
    export = pd.DataFrame({'S':np.around(s_vec,0), 'I': np.around(i_vec,0), 'R':np.around(r_vec+m_vec,0), 'sintomatici_modello':np.around(i_vec/3,0)})
    export.index = new_t

    # Data dataframe
    new_ydata_inf = pd.DataFrame({'sintomatici_data':np.around(ydata_inf,0)})
    new_ydata_inf.index = data_t

    # Join and export
    joint_frames = export.join(new_ydata_inf,on=export.index)
    export2 = joint_frames.iloc[:200,:]
    export2.index.name='data'
    export2.to_csv('output/'+fit_region+'.csv',index=True)


# %%



# %%


