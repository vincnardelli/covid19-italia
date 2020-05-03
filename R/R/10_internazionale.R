#internazionale

# input =  R0 all in one

data <- read_csv("fit_modelli/r0_countries.csv")

# Grafico a discesa (aggiungere area per filtro) 
data <- read_csv("fit_modelli/r0_countries.csv")
data   %>% rename("Data" = X1) %>% filter(Data > "2020-03-10") %>% write.xlsx(paste0("export/10_R0_europe_curve_",format(Sys.Date(), "%d%m"),".xlsx"))

# grafico a discesa animato
data <- data  %>% rename("Data" = X1) %>%  filter(Data > "2020-03-10") 
gather(data, key = Nazione, value = r0 , -Data) %>% 
  pivot_wider(names_from = Data, values_from = r0) %>% write.xlsx(paste0("export/10_R0_europe_animated_",format(Sys.Date(), "%d%m"),".xlsx"))


# istogramma R0
data <- read_csv("fit_modelli/r0_countries.csv")
data <- data  %>% rename("Data" = X1) %>%  filter(Data == max(Data)) 
data <-  gather(data, key = NAME, value = r0 , -Data) %>% 
  pivot_wider(names_from = Data, values_from = r0) 
names(data)[2] <- "R0"
data %>%filter(R0 != 0, NAME %in% c("Italy", "Germany","France","Spain", "United Kingdom","Sweden","Austria","Belgium","Netherlands","Serbia","Slovakia","Slovenia","Monaco","Ukraine","Turkey","Greece","Poland","Portugal")) %>% arrange (desc(R0)) %>%
  write.xlsx(paste0("export/10_R0_europe_hist_",format(Sys.Date(), "%d%m"),".xlsx"))

data$NAME[data$NAME == "North Macedonia"] <- "Macedonia"
data$NAME[data$NAME == "Czechia"] <- "Czech Republic"



#mappa R0 europea
#map <- geojson_read("data/custom.geo.json",  what = "sp")
map <- geojson_read("https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson", what = "sp")


levels(map@data$NAME)[48] <- "Macedonia"
levels(map@data$NAME)[38] <- "Moldova"

map@data <- map@data %>% 
  left_join(data) #%>% filter(NAME != "Israel") 
names(map@data)[13] <- "R0"

# ultimo aggiornamento 

last <- read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-23-2020.csv")
last <- aggregate(last[,8:11], by= list(as.factor(last$Country_Region)), FUN = "sum")
last <- last %>% rename("NAME" = Group.1,"Totale casi" = Confirmed,"Guariti" = Recovered,"Deceduti" = Deaths, "Attualmente positivi" = Active) %>%
  mutate("Letalità" = round(Deceduti/`Totale casi`,2),
         "Guariti/Deceduti" = round(Guariti/Deceduti,2))
levels(last$NAME)[46] <- "Czech Republic"
levels(last$NAME)[126] <- "Macedonia"
#write.xlsx(last,"export/10_dati_odierni.xlsx")
map@data <- map@data %>% left_join(last) 
#data %>% left_join(last) %>% View()
geojson_write(map, file = paste0("export/10_mappa_R0_europa_",format(Sys.Date(), "%d%m"),".geojson"))




#italia

#R0
nation <- read_csv("fit_modelli/r0.csv") 
# numeri
italia <- read_csv("https://github.com/pcm-dpc/COVID-19/raw/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
italia <- italia %>% select(data,totale_casi, totale_positivi, dimessi_guariti,deceduti)
italia <- italia %>% filter(data == max(data)) %>%  rename("Data" = data, "Totale casi" = totale_casi, "Attualmente positivi" = totale_positivi, "Guariti" = dimessi_guariti, "Deceduti" = deceduti) %>% mutate("Letalità" = round(Deceduti/`Totale casi`,2),
           "Guariti/Deceduti" = round(Guariti/Deceduti,2))
nation <- nation %>% filter(Data == max(Data))
italia <- cbind(R0 = nation$R0,  italia[,-1])
  
#italia <- cbind(data.frame(Data = as_date(italia[-1,]$data),"Totale casi" = italia[-1,]$totale_casi, "Incremento positivi" = italia[-1,]$variazione_totale_positivi ), data.frame(diff(as.matrix(italia[,-c(1,2)]), lag = 1, differences = 1))) 
#italia <- italia %>% rename("Incremento positivi" = Attualmente.positivi, "Guariti" = dimessi_guariti, "Deceduti" = deceduti)



# Andamento totale casi
Totale <- read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
Totale$`Country/Region`[Totale$`Country/Region` == "Czechia"] <- "Czech Republic"
Totale$`Country/Region`[Totale$`Country/Region` == "North Macedonia"] <- "Macedonia"
Totale <- Totale %>% filter(`Country/Region` %in% map@data$NAME) %>% select(-c(`Province/State`,Lat,Long))
Totale <- aggregate(Totale[,-1], by= list(as.factor(Totale$`Country/Region`)), FUN = "sum") 


gather(Totale, key = Data, value = Totale, -Group.1) %>% 
  pivot_wider(names_from = Group.1, values_from = Totale) %>% 
  write.xlsx(paste0("export/10_tot_casi_europe_",format(Sys.Date(), "%d%m"),".xlsx"))




# Variazione giornaliera infetti (curve)
#incremento <- gather(Totale, key = Data, value = Totale, -Group.1) %>% 
#  pivot_wider(names_from = Group.1, values_from = Totale)
#incremento <-  cbind(data.frame(Data = mdy(incremento[-1,]$Data)), data.frame(diff(as.matrix(incremento[,-1]), lag = 1, differences = 1))) 
#
#
#write.xlsx(incremento,"export/10_variazione_giornaliera_europe.xlsx")
#
## Variazione giornaliera infetti (hist)
#incremento <- incremento %>% filter(as_date(Data) == max(as_date(Data))) 
#gather(incremento, key = Paese, value = "Attualmente positivi", -Data) %>% write.xlsx("export/10_variazione_giornaliera_europe_hist.xlsx")

#Andamento Guariti
Guariti <- read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
Guariti$`Country/Region`[Guariti$`Country/Region` == "Czechia"] <- "Czech Republic"
Guariti$`Country/Region`[Guariti$`Country/Region` == "North Macedonia"] <- "Macedonia"
Guariti <- Guariti %>% filter(`Country/Region` %in% map@data$NAME) %>% select(-c(`Province/State`,Lat,Long))
Guariti <- aggregate(Guariti[,-1], by= list(as.factor(Guariti$`Country/Region`)), FUN = "sum") 
gather(Guariti, key = Data, value = Guariti, -Group.1) %>% 
  pivot_wider(names_from = Group.1, values_from = Guariti) %>% 
  write.xlsx(paste0("export/10_guariti_europe_",format(Sys.Date(), "%d%m"),".xlsx"))



#Andamento Deceduti
Deceduti <- read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
Deceduti$`Country/Region`[Deceduti$`Country/Region` == "Czechia"] <- "Czech Republic"
Deceduti$`Country/Region`[Deceduti$`Country/Region` == "North Macedonia"] <- "Macedonia"
Deceduti <- Deceduti %>% filter(`Country/Region` %in% map@data$NAME) %>% select(-c(`Province/State`,Lat,Long))
Deceduti <- aggregate(Deceduti[,-1], by= list(as.factor(Deceduti$`Country/Region`)), FUN = "sum") 
gather(Deceduti, key = Data, value = Deceduti, -Group.1) %>% 
  pivot_wider(names_from = Group.1, values_from = Deceduti) %>% 
  write.xlsx(paste0("export/10_deceduti_europe_",format(Sys.Date(), "%d%m"),".xlsx"))


# Variazione giornaliera infetti
T <- gather(Totale, key = Data, value = Totale, -Group.1)
G <- gather(Guariti, key = Data, value = Guariti, -Group.1)
D <- gather(Deceduti, key = Data, value = Deceduti, -Group.1)

all <- cbind(T, Guariti = G$Guariti,Deceduti = D$Deceduti) 
all$Data <- mdy(all$Data)
all <- all %>% filter(Data > "2020-02-22") %>%  mutate("Attualmente positivi" = Totale - Guariti - Deceduti,
               "Letalità" = round(Deceduti/Totale,2),
               "Guariti/Deceduti" = round(Guariti/Deceduti,2)) 
all <- all %>% select(Group.1,Data,`Attualmente positivi`) %>%  pivot_wider(names_from = Group.1, values_from = `Attualmente positivi`)
write.xlsx(all,paste0("export/10_attualmente_positivi_europe_",format(Sys.Date(), "%d%m"),".xlsx"))

all <-  cbind(data.frame(Data = all[-1,]$Data), data.frame(diff(as.matrix(all[,-1]), lag = 1, differences = 1))) 
write.xlsx(all,paste0("export/10_variazione_giornaliera_europe_",format(Sys.Date(), "%d%m"),".xlsx"))

# Variazione giornaliera infetti (hist)
all <- all %>% filter(as_date(Data) == max(as_date(Data))) 
gather(all, key = Paese, value = "Attualmente positivi", -Data) %>% write.xlsx(paste0("export/10_variazione_giornaliera_europe_hist_",format(Sys.Date(), "%d%m"),".xlsx"))





# input = R0 scomposto
#
#nations <- list("Italy", "France","Germany", "Spain","UK","USA", "Sweden")
#
#data <- data.frame(Data = seq(as_date("2020-01-26"),as_date("2020-04-16"),by = "days"))
#for (i in (1:length(nations))) {
#  nation <- read_csv(paste0("internazionale/R0_",nations[[i]],".csv")) %>% select(R0)
#  names(nation)[1] <- nations[[i]]
#  data <- cbind(data,nation)
#}
#
#data %>% filter(Data > "2020-03-02") %>% gather(Italy, France,Germany, Spain,UK,USA, key = Area, value = r0 ) %>% 
#  pivot_wider(names_from = Data, values_from = r0) %>% 
#  write.xlsx("export/10_R0_internazionale.xlsx")
#
#Europe <- data.frame("Paese"= c("Armenia","Azerbaijan","San Marino","Albania","Andorra","Austria","Belgium","Bulgaria","Bosnia and Herzegovina","Belarus",
#                                "Switzerland","Northern Cyprus","Cyprus","Czechia","Germany","Denmark","Spain","Estonia","Finland","France",
#                                "United Kingdom","Georgia","Greece","Croatia","Hungary","Ireland","Iceland","Italy","Kosovo","Liechtenstein",
#                                "Lithuania","Luxembourg","Latvia","Moldova","North Macedonia","Malta","Montenegro","Netherlands","Norway","Poland",
#                              "Romania","Republic of Serbia","Slovakia","Slovenia","Sweden","Turkey","Ukraine","Portugal"))
