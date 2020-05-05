#SIR NAZIONALE
library(readr)
library(dplyr)
library(openxlsx)
library(lubridate)

data <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
data$data <- as_date(data$data)

#EXPORT


export <- read_csv("fit_modelli/nazionale.csv")
export$data <- seq(ymd('2020-02-24'),ymd('2021-02-22'), by = '1 day')
export$data <- as_date(export$data)



export <- export  %>% 
  mutate(sint = i/3) %>% 
  left_join(data) %>% 
  select(data, s, i, r, sint, totale_positivi)%>%  
  rename(Sucettibili = s,
                                                                      Infetti = i,
                                                                      Rimossi = r,
                                                                      Sintomatici = sint,
                                                                      "Casi Attuali" = totale_positivi)
export %>% filter(data > "2020-02-23", data <= "2020-08-10") %>% 
write.csv(paste0("export/5_sir_nazionale_", format(Sys.Date(), "%d%m"),".csv"))

# predizione R0
export <- export %>% filter(data > "2020-02-23", data <= "2020-10-15")

pred <- read_csv("fit_modelli/predizioni_future_export.csv") %>%  filter(date > "2020-02-23", date <= "2020-10-15")
pred <- cbind(pred, "Proiezione ad oggi" = export$Sintomatici)
pred$date <- as_date(pred$date)
pred %>% rename("R0 tra 1.2 e 1.1" = `R0=1.2`,
                "R0 tra 1.1 e 1.0" = `R0=1.1`,
                "R0 tra 1.0 e 0.5" = `R0=1`,
                "Casi reali" = infetti_reali) %>% 
  select(date,`R0 tra 1.2 e 1.1`, `R0 tra 1.1 e 1.0`, `R0 tra 1.0 e 0.5`,`Proiezione ad oggi`, `Casi reali`) %>% 
write.csv(paste0("export/5_proiezione_R0", format(Sys.Date(), "%d%m"),".csv"))
#GRAFICO INCREMENTO

incremento <- export %>% mutate(lag_sint = lag(Sintomatici),
                                lag_att = lag(`Casi Attuali`),
                                "Variazione prevista" = Sintomatici - lag_sint,
                                "Variazione reale" = `Casi Attuali`- lag_att) %>% 
  select(data,`Variazione prevista`,`Variazione reale`) %>% filter(data <= "2020-05-10")


write.csv(incremento, paste0("export/5_incremento_", format(Sys.Date(), "%d%m"),".csv"))



# INCREMENTO GIORNALIERO REGIONALE

regioni_latest <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-latest.csv")
regioni_latest$data <- as_date(regioni_latest$data)
#regioni_latest$denominazione_regione <-  gsub("P.A. Bolzano", "Trentino Alto Adige", regioni_latest$denominazione_regione)
#regioni_latest$denominazione_regione <- gsub("P.A. Trento", "Trentino Alto Adige", regioni_latest$denominazione_regione)

regioni_latest %>% select(denominazione_regione,variazione_totale_positivi) %>% 
  mutate(zona = case_when(denominazione_regione %in% c("Sardegna", "Sicilia") ~ "ISOLE",
                          denominazione_regione %in% c("Basilicata", "Calabria","Campania","Puglia","") ~ "SUD",
                          denominazione_regione %in% c("Abruzzo", "Campania","Lazio","Marche","Molise","Toscana","Umbria") ~ "CENTRO",
                          denominazione_regione %in% c("P.A. Bolzano", "Emilia-Romagna","Friuli Venezia Giulia","Liguria","Lombardia","Piemonte","P.A. Trento","Valle d'Aosta","Veneto") ~ "NORD")) %>% arrange(desc(variazione_totale_positivi)) %>%
  rename("Regione" = denominazione_regione, "Variazione positivi" = variazione_totale_positivi, "Area" = zona) %>% 
  write.xlsx(paste0("export/incremento_regionale_",format(Sys.Date(), "%d%m"),".xlsx"))





