import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
from datetime import date, timedelta
import geopandas as gpd

#import today date
date_today = date.today()
year_t,month_t,date_t=str(date_today).split('-')

# The SIR model differential equations.
def deriv(y, t, N, beta,gamma):
    S,I,R = y

    dSdt = -(beta*I/N)*S 
    dIdt = (beta*S/N)*I - gamma*I 
    dRdt = gamma*I 
    
    return dSdt, dIdt, dRdt


#Integration of the differential equations
    
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

countries_list=['Albania',
                'Armenia',
                'Austria',
                'Azerbaijan',
                'Belarus',
                'Belgium',
                'Bosnia and Herzegovina',
                'Bulgaria',
                'Cyprus',
                'Croatia',
                'Czechia',
                'Denmark',
                'Estonia',
                'Finland',
                'France',
                'Georgia',
                'Germany',
                'Greece',
                'Hungary',
                'Iceland',
                'Ireland',
                'Israel',
                'Italy',
                'Kazakhstan',
                'Kyrgyzstan',
                'Latvia',
                'Lithuania',
                'Luxembourg',
                'Malta',
                'Moldova',
                'Monaco',
                'Montenegro',
                'Netherlands',
                'North Macedonia',
                'Norway',
                'Poland',
                'Portugal',
                'Romania',
                'Serbia',
                'Slovakia',
                'Slovenia',
                'Spain',
                'Sweden',
                'Switzerland',
                'Turkey',
                'Ukraine',
                'United Kingdom']

#IMPORT FILES WORLD
#i files sono: le righe sono le nazioni, le colonne i giorni del mese (DATE).

file_confirmed='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
file_deaths='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
file_recovered='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

df_confirmed=pd.read_csv(file_confirmed)
df_deaths=pd.read_csv(file_deaths)
df_recovered=pd.read_csv(file_recovered)

countries_w_confirmed = df_confirmed['Country/Region']
countries_w_deaths = df_deaths['Country/Region']
countries_w_recovered = df_recovered['Country/Region']

#confirmed world
confirmed_world0 = df_confirmed.drop(['Province/State','Lat','Long'], 
                                    axis=1)
confirmed_world0.rename(index=countries_w_confirmed, inplace=True)
confirmed_world = confirmed_world0.drop(['Country/Region'], 
                                        axis=1).T.reset_index()
confirmed_world.rename(columns={'index':'Date'}, inplace=True)

#deaths world
deaths_world0 = df_deaths.drop(['Province/State','Lat','Long'], 
                                    axis=1)
deaths_world0.rename(index=countries_w_deaths, inplace=True)
deaths_world = deaths_world0.drop(['Country/Region'], 
                                        axis=1).T.reset_index()
deaths_world.rename(columns={'index':'Date'}, inplace=True)

#recovered world
recovered_world0 = df_recovered.drop(['Province/State','Lat','Long'], 
                                    axis=1)
recovered_world0.rename(index=countries_w_recovered, inplace=True)
recovered_world = recovered_world0.drop(['Country/Region'], 
                                        axis=1).T.reset_index()
recovered_world.rename(columns={'index':'Date'}, inplace=True)

confirmed_europe0 = confirmed_world[countries_list]
deaths_europe0 = deaths_world[countries_list]
recovered_europe0 = recovered_world[countries_list]

array_names=([])
for name in countries_list:
    array_names.append([name,list(countries_w_confirmed).count(name)])

Totale=pd.DataFrame()
for i in range(0, len(countries_list)):
    if array_names[i][1] > 1:
               Totale.insert(i, 
                              countries_list[i], 
                              value=confirmed_europe0[countries_list[i]].T.sum())
    elif array_names[i][1]==1:
                Totale.insert(i, 
                                countries_list[i], 
                                value=confirmed_europe0[countries_list[i]].T)

Totale.insert(0, 'Date', confirmed_world['Date'])

Deceduti=pd.DataFrame()
for i in range(0, len(countries_list)):
    if array_names[i][1] > 1:
            Deceduti.insert(i, 
                              countries_list[i], 
                              value=deaths_europe0[countries_list[i]].T.sum())
    elif array_names[i][1]==1:
            Deceduti.insert(i, 
                              countries_list[i], 
                              value=deaths_europe0[countries_list[i]].T)

Deceduti.insert(0, 'Date', deaths_world['Date'])

Guariti=pd.DataFrame()
for i in range(0, len(countries_list)):
    if array_names[i][1] > 1:
                Guariti.insert(i, 
                              countries_list[i], 
                              value=recovered_europe0[countries_list[i]].T.sum())
    elif array_names[i][1]==1:
                Guariti.insert(i, 
                                 countries_list[i], 
                                 value=recovered_europe0[countries_list[i]].T)

Guariti.insert(0, 'Date', recovered_world['Date'])

#Active Infected
Attualmente_positivi=pd.DataFrame()

for i in range(0, len(countries_list)):
    Attualmente_positivi.insert(i, 
                                  countries_list[i], 
                                  value=
                                     Totale[countries_list[i]]-
                                      Deceduti[countries_list[i]]-
                                      Guariti[countries_list[i]])

Attualmente_positivi.insert(0, 'Date', confirmed_world['Date'])

Totale.to_csv('output/10_tot_casi_europe_'+date_t+month_t+'.csv', index=True)
Deceduti.to_csv('output/10_deceduti_europe_'+date_t+month_t+'.csv', index=True)
Guariti.to_csv('output/10_guariti_europe_'+date_t+month_t+'.csv', index=True)
Attualmente_positivi.to_csv('output/10_attualmente_positivi_europe_'+date_t+month_t+'.csv', index=True)

#Daily variation infected
Variazione_giornaliera = pd.DataFrame(Attualmente_positivi['Date'].iloc[1:])

for name in countries_list:
    active_var=([])    
    for i in range(1,len(Attualmente_positivi)):
        active_var.append(Attualmente_positivi[name][i]-Attualmente_positivi[name][i-1])
    Variazione_giornaliera[name]=active_var

Variazione_giornaliera.to_csv('output/10_variazione_giornaliera_europe_'+date_t+month_t+'.csv', index=True)

def func_plot(df):
    
    y_world=[]
    n_cols=df.shape[1]
    
    for i in range(n_cols-4):
        y_world.append(df.iloc[:,i+4].sum())
    
    x_world2=df.columns[4:]
    x_world=pd.to_datetime(x_world2,infer_datetime_format=False)
    
    return (x_world,y_world)

#Generalization to other countries

def whichcountry(name):

    ######## INPUT PARAMETERS ########
    country=name
    t0=pd.to_datetime('2020-01-22')
    #################################

    mask_coun=df_confirmed['Country/Region']==country   # you can change the country here
    mask_coun_rec=df_recovered['Country/Region']==country

    df_confirmed_C=df_confirmed.loc[mask_coun,:]
    df_deaths_C=df_deaths.loc[mask_coun,:]
    df_recovered_C=df_recovered.loc[mask_coun_rec,:]

    ytot=np.array(func_plot(df_confirmed_C)[1])
    ydeaths=np.array(func_plot(df_deaths_C)[1])
    yrec=np.array(func_plot(df_recovered_C)[1])

    return ytot-ydeaths-yrec, ytot[-1], yrec[-1],ydeaths[-1]

xdata=pd.to_numeric(range(Attualmente_positivi.shape[0]))

today=len(xdata)

def minimizer(R0,t1=today-5,t2=today):
    array_country_bis=array_country
    
    #true data
    ydata_inf_2=array_country[t1:t2]
    xdata_2=np.arange(0,len(ydata_inf_2))
    
    #model
    fin_result=time_evo(60*10**6,1/14*R0,1/14,I0=ydata_inf_2[0])
    i_vec=fin_result[2]
    i_vec_2=i_vec[0:len(xdata_2)]
    
    #average error
    error=np.sum(np.abs(ydata_inf_2-i_vec_2)/ydata_inf_2)*100
    return error

minimizer_vec=np.vectorize(minimizer)

time_window=5

def minimizer_gen(t1,t2,xgrid=np.arange(0.1,5,0.01)):

    ygrid=minimizer_vec(xgrid,t1=t1,t2=t2)
    r0_ideal=round(xgrid[np.argmin(ygrid)],2)

    return r0_ideal

r0_today=[]
scangrid=np.linspace(0,3,400)
name_R0_array = []

for name in range(0, len(countries_list)):
    
    array_country=whichcountry(countries_list[name])[0]
    
    i = today-(time_window-1)
    min_today=minimizer_gen(i,i+time_window,scangrid)
    r0_today.append(min_today)
    #scangrid=np.linspace(0,5,200)
    name_R0_array.append([countries_list[name], min_today])

name_R0_df = pd.DataFrame(name_R0_array, columns=['Country', 'R0'])

countries_hist=['United Kingdom',
                'Ukraine',
                'Poland',
                'Greece',
                'Netherlands',
                'Portugal',
                'Belgium',
                'France',
                'Slovenia',
                'Serbia',
                'Spain',
                'Italy',
                'Sweden',
                'Austria',
                'Slovakia',
                'Turkey']

hist_list=[]
for i in range(len(countries_hist)):
    ind = name_R0_df.loc[name_R0_df['Country'] == countries_hist[i]].index[0]
    hist_list.append([name_R0_df['Country'][ind], name_R0_df['R0'][ind]])
hist_df = pd.DataFrame(hist_list, columns=['Country', 'R0'])
hist_df.to_csv('output/10_R0_europe_hist_'+date_t+month_t+'.csv')

#import yesterday date
yesterday = date.today() - timedelta(days=1)
year_y,month_y,date_y=str(yesterday).split('-')

r0_countries_imp = pd.read_excel('input/input.xlsx')
r0_imp_noindex = r0_countries_imp.iloc[:, 1:]

row_today=pd.DataFrame(np.reshape(r0_today,(1, len(countries_list))),
                       index= [str(date.today())],
                       columns=countries_list).reset_index()
row_today.rename(columns={'index':'Date'}, inplace=True)
row_today.index = [len(r0_imp_noindex)]

export_today = pd.concat([r0_imp_noindex,row_today])
export_today.to_excel('output/10_R0_europe_curve_'+date_t+month_t+'.xlsx',index=True)
export_today.to_excel('input/input.xlsx',index=True)

r0_to_join = pd.Series(name_R0_df['R0'])
r0_to_join.index = name_R0_df['Country']
confirmed_to_join = Totale.iloc[-1, 1:]
deaths_to_join = Deceduti.iloc[-1, 1:]
recovered_to_join = Guariti.iloc[-1, 1:]
ai_to_join = Attualmente_positivi.iloc[-1, 1:]

frame = {'R0':r0_to_join,
         'Confirmed': confirmed_to_join, 
         'Deaths': deaths_to_join, 
         'Recovered':recovered_to_join, 
         'Active Infected': ai_to_join}

df_to_join = pd.DataFrame(frame)
df_to_join.rename(index={'Czechia':'Czech Republic', 
                         'Moldova':'Republic of Moldova', 
                         'North Macedonia':'The former Yugoslav Republic of Macedonia'})
df_to_join.reset_index()

#Map
map = gpd.read_file("https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson")
map = map.join(df_to_join, on='NAME', how='left')
map.to_file('output/10_mappa_R0_europa_'+date_t+month_t+'.geojson', driver='GeoJSON')