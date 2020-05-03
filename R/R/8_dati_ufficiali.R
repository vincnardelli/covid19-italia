library(readxl)
library(lubridate)
library(dplyr)
library(openxlsx)
library(stringr)

#DATI UFFICIALI DIPARTIMENTO PROTEZIONE CIVILE 

#NAZIONALI
nazione <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
nazione$data <- substr(nazione$data,1,10)
nazione$data <- as_date(nazione$data)


nazione %>% arrange(desc(data)) %>% rename("Data" = data,
                                           "Ricoverati con sintomi" = ricoverati_con_sintomi,
                                                      "Terapia intensiva" = terapia_intensiva,
                                                      "Isolamento domiciliare" = isolamento_domiciliare,
                                                      "Attualmente positivi" = totale_positivi,
                                                      "Variazione positivi" = variazione_totale_positivi,
                                                      "Guariti" = dimessi_guariti,
                                                      "Deceduti" = deceduti,
                                                      "Tamponi" = tamponi
                                                      ) %>% select(Data,`Attualmente positivi`,`Variazione positivi`,`Terapia intensiva`,
                              `Ricoverati con sintomi`,`Isolamento domiciliare`,Guariti,Deceduti,Tamponi) %>% 
  write.xlsx(paste0("export/8_nazionale_", format(Sys.Date(), "%d%m"),".xlsx"))

#REGIONALI
regioni <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
regioni$data <- as_date(regioni$data)


regioni %>% arrange(desc(data)) %>% mutate(Area = case_when(denominazione_regione %in% c("Sardegna", "Sicilia") ~ "Isole",
                                                            denominazione_regione %in% c("Basilicata", "Calabria","Campania","Puglia","") ~ "Sud",
                                                            denominazione_regione %in% c("Abruzzo", "Campania","Lazio","Molise","Toscana","Umbria") ~ "Centro",
                                                            denominazione_regione %in% c("P.A. Bolzano", "Emilia-Romagna","Friuli Venezia Giulia","Liguria","Lombardia","Marche","Piemonte","P.A. Trento","Valle d'Aosta","Veneto") ~ "Nord")) %>% 
  rename("Data" = data,
                                           "Regione" = denominazione_regione,
                                           "Ricoverati con sintomi" = ricoverati_con_sintomi,
                                           "Terapia intensiva" = terapia_intensiva,
                                           "Isolamento domiciliare" = isolamento_domiciliare,
                                           "Attualmente positivi" = totale_positivi,
                                           "Variazione positivi" = variazione_totale_positivi,
                                           "Guariti" = dimessi_guariti,
                                           "Deceduti" = deceduti,
                                           "Tamponi" = tamponi
) %>% select(Data,Area,Regione, `Attualmente positivi`,`Variazione positivi`,`Terapia intensiva`,
             `Ricoverati con sintomi`,`Isolamento domiciliare`,Guariti,Deceduti, Tamponi) %>% 
  write.xlsx(paste0("export/8_regionale_", format(Sys.Date(), "%d%m"),".xlsx"))

#PROVINCIALI
province <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv")
province$data <- as_date(province$data)
province <- province %>% filter(denominazione_provincia != "In fase di definizione/aggiornamento")

province$denominazione_regione <-  gsub("P.A. Bolzano", "Trentino Alto Adige", province$denominazione_regione)
province$denominazione_regione <- gsub("P.A. Trento", "Trentino Alto Adige", province$denominazione_regione)

province <- province %>% arrange(desc(data)) %>% rename("Data" = data,
                                                        "Regione" = denominazione_regione,
                                                        "Provincia" = denominazione_provincia,
                                                        "Totale casi" = totale_casi) %>% 
  select(Data, Regione, Provincia, `Totale casi`) %>% 
  write.xlsx(paste0("export/8_provinciale_", format(Sys.Date(), "%d%m"),".xlsx"))



#COUNTERS

counters <- nazione %>% filter(data == max(data)) %>% select(totale_positivi,dimessi_guariti,deceduti,totale_casi) %>% 
  rename("Positivi attuali" = totale_positivi,
         "Guariti" = dimessi_guariti,
         "Deceduti" = deceduti,
         "Totale casi" = totale_casi)
counters_regioni <- regioni %>% filter(data == max(data)) %>%  mutate(Area = case_when(denominazione_regione %in% c("Sardegna", "Sicilia") ~ "Isole",
                                                                                       denominazione_regione %in% c("Basilicata", "Calabria","Campania","Puglia","") ~ "Sud",
                                                                                       denominazione_regione %in% c("Abruzzo", "Campania","Lazio","Molise","Toscana","Umbria") ~ "Centro",
                                                                                       denominazione_regione %in% c("P.A. Bolzano", "Emilia-Romagna","Friuli Venezia Giulia","Liguria","Lombardia","Marche","Piemonte","P.A. Trento","Valle d'Aosta","Veneto") ~ "Nord")) %>% 
  select(Area, denominazione_regione,totale_positivi,dimessi_guariti,deceduti,totale_casi) 

write.xlsx(counters_regioni,paste0("export/counter_regioni",format(Sys.Date(), "%d%m"),".xlsx"))



#dati sardegna
#sardegna <- regioni %>% filter(denominazione_regione == "Sardegna") %>% select(data,totale_positivi) %>% mutate(giorni = c(1:nrow(sardegna)))
#write.xlsx(sardegna,"export/sardegna.xlsx")

#regioni %>% filter( data == max(data)) %>% select(denominazione_regione,tamponi) %>% arrange(desc(tamponi))


