# Tamponi vs positivi

#nazionali
naz <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
naz$data <- as_date(naz$data)

naz %>% select(data,nuovi_positivi,tamponi) %>% mutate(lag_tamp = lag(tamponi),
                                                       tamponi_giornalieri = tamponi - lag_tamp,
                                                       "% positivi su tamponi" = nuovi_positivi/tamponi_giornalieri*100) %>% 
  rename("Nuovi contagi giornalieri" = nuovi_positivi,
         "Tamponi giornalieri" = tamponi_giornalieri) %>% select(data,"Nuovi contagi giornalieri", "Tamponi giornalieri", "% positivi su tamponi") 
  #write.xlsx("export/tamponi_naz.xlsx")





#regionali
reg <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
reg$data <- as_date(reg$data)

reg %>% select(data,denominazione_regione,totale_positivi,tamponi) %>% mutate("% positivi su tamponi" = totale_positivi/tamponi*100) %>% 
  rename(Regione = denominazione_regione,
         Positivi = totale_positivi,
         Tamponi = tamponi) %>% 
  na_if(.,0) %>% write.xlsx("export/tamponi_reg.xlsx")


reg %>% select(data,denominazione_regione,nuovi_positivi,tamponi)%>% group_by(denominazione_regione,data) %>%  mutate(lag_tamp = lag(tamponi),
                                                                            tamponi_giornalieri = tamponi - lag_tamp,
                                                                            "% positivi su tamponi" = nuovi_positivi/tamponi_giornalieri*100) %>% 
  rename("Nuovi contagi giornalieri" = nuovi_positivi,
         "Tamponi giornalieri" = tamponi_giornalieri) %>% select(data,denominazione_regione,"Nuovi contagi giornalieri", "Tamponi giornalieri", "% positivi su tamponi") 
#write.xlsx("export/tamponi_reg.xlsx")
