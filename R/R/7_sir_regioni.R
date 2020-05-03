#SIR GEOGRAFICO
library(readr)
library(dplyr)
library(openxlsx)
library(lubridate)

# Numeri attuali
regioni_latest <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-latest.csv")
regioni_latest$data <- as_date(regioni_latest$data)
zone <- regioni_latest %>% mutate(data = format(data, "%d-%m-%Y"),
                           zona = case_when(denominazione_regione %in% c("Sardegna", "Sicilia") ~ "Isole",
                                            denominazione_regione %in% c("Basilicata", "Calabria","Campania","Puglia","") ~ "Sud",
                                            denominazione_regione %in% c("Abruzzo", "Campania","Lazio","Marche","Molise","Toscana","Umbria") ~ "Centro",
                                            denominazione_regione %in% c("P.A. Bolzano", "Emilia-Romagna","Friuli Venezia Giulia","Liguria","Lombardia","Piemonte","P.A. Trento","Valle d'Aosta","Veneto") ~ "Nord")) %>% 
  group_by(zona) %>% summarise(sintomatici = sum(totale_positivi),
                               deceduti = sum(deceduti)) %>% arrange(desc(sintomatici))
print(zone)




#EXPORT NORD
export_Nord <- read_csv("fit_modelli/nord.csv")
export_Nord <- export_Nord  %>% filter(data > "2020-02-23", data < "2020-08-10") %>%  rename(Sucettibili = S,
                                                                      Infetti = I,
                                                                      Rimossi = R,
                                                                      Previsti = sintomatici_modello,
                                                                      "Casi Attuali" = sintomatici_data)

write.csv(export_Nord, paste0("export/7_sir_nord_", format(Sys.Date(), "%d%m"),".csv"))

#EXPORT CENTRO
export_Centro <- read_csv("fit_modelli/centro.csv")
export_Centro <- export_Centro %>% filter(data > "2020-02-23", data < "2020-07-25") %>% rename(Sucettibili = S,
                                                                      Infetti = I,
                                                                      Rimossi = R,
                                                                      Previsti = sintomatici_modello,
                                                                      "Casi Attuali" = sintomatici_data)

write.csv(export_Centro, paste0("export/7_sir_centro_", format(Sys.Date(), "%d%m"),".csv"))

#EXPORT SUD
export_Sud <- read_csv("fit_modelli/sud.csv")
export_Sud <- export_Sud %>% filter(data > "2020-02-23", data < "2020-07-25") %>% rename(Sucettibili = S,
                                                                      Infetti = I,
                                                                      Rimossi = R,
                                                                      Previsti = sintomatici_modello,
                                                                      "Casi Attuali" = sintomatici_data)

write.csv(export_Sud, paste0("export/7_sir_sud_", format(Sys.Date(), "%d%m"),".csv"))

#EXPORT ISOLE
export_Isole <- read_csv("fit_modelli/isole.csv")
export_Isole <- export_Isole  %>% filter(data > "2020-02-23", data < "2020-08-16") %>% rename(Sucettibili = S,
                                                                      Infetti = I,
                                                                      Rimossi = R,
                                                                      Previsti = sintomatici_modello,
                                                                      "Casi Attuali" = sintomatici_data)
write.csv(export_Isole, paste0("export/7_sir_isole_", format(Sys.Date(), "%d%m"),".csv"))

