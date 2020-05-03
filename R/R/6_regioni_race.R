# Grafico animato
#REGIONALI
regioni <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
regioni$data <- as_date(regioni$data)
regioni %>% 
  mutate(data = format(data, "%d-%m-%Y"),
         zona = case_when(denominazione_regione %in% c("Sardegna", "Sicilia") ~ "Isole",
                          denominazione_regione %in% c("Basilicata", "Calabria","Campania","Puglia","") ~ "Sud",
                          denominazione_regione %in% c("Abruzzo", "Campania","Lazio","Marche","Molise","Toscana","Umbria") ~ "Centro",
                          denominazione_regione %in% c("P.A. Bolzano", "Emilia-Romagna","Friuli Venezia Giulia","Liguria","Lombardia","Piemonte","P.A. Trento","Valle d'Aosta","Veneto") ~ "Nord")) %>% 
  select(zona,data, totale_positivi, denominazione_regione) %>% 
  pivot_wider(names_from = data, values_from = totale_positivi) %>% arrange(zona) %>% 
  write.xlsx(paste0("export/6_regionirace_", format(Sys.Date(), "%d%m"),".xlsx"))

