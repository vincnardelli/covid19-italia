library(writexl)
library(dplyr)
library(geojsonio)
library(ggplot2)
data <- read.csv("https://github.com/pcm-dpc/COVID-19/raw/master/dati-regioni/dpc-covid19-ita-regioni.csv", stringsAsFactors = FALSE)
ti <- readxl::read_xlsx("data/nuovi_posti_TI.xlsx")


lista <- list(
  list(
    nome="nord", 
    regioni=c("P.A. Bolzano", 
      "P.A. Trento",
      "Valle d'Aosta",
      "Veneto", 
      "Piemonte",
      "Lombardia",
      "Liguria", 
      "Friuli-Venezia Giulia", 
      "Emilia-Romagna")),
  list(
    nome="centro", 
    regioni=c("Abruzzo",
              "Marche",
              "Molise",
              "Lazio",
              "Toscana", 
              "Umbria")),
  list(
    nome="sud", 
    regioni= c("Basilicata", 
               "Calabria",
               "Campania",
               "Puglia")),
  list(
    nome="isole", 
    regioni=c("Sardegna", 
              "Sicilia"))
  
)




for(i in 1:length(lista)){
  
  nome <- lista[[i]]$nome
  regioni <- lista[[i]]$regioni
  
  ti_df <- data %>% 
    filter(denominazione_regione %in% regioni) %>% 
    group_by(data) %>% 
    summarise(ti=sum(terapia_intensiva))
  
  ti_df$data <- as_date(ti_df$data)
  model <- read.csv(paste0("fit_modelli/", nome, ".csv"))
  model$data <- as_date(model$data)
  
  final <- right_join(ti_df, model) %>% 
    filter(data > "2020-02-24" & data < "2020-06-01")
  
  
  ti_lines <- ti %>% 
    filter(regione %in% regioni) 
  ti_lines <- colSums(ti_lines[, 2:3])
  
  df <- data.frame("data"=final$data,
                   "A"=final$sintomatici_modello*0.05,
                   "B"=final$sintomatici_modello*0.07,
                   "C"=final$ti)
  df <- df %>% rename("Data" = data,
                      "5% dei sintomatici" = A,
                      "7% dei sintomatici" = B,
                      "Pazienti attualmente in terapia intensiva" = C)
  
  write_xlsx(df, paste0("export/4_tiregioni_",nome, "_", format(Sys.Date(), "%d%m"),".xlsx"))
  
}


# Soglie
s_nord <- ti %>% filter(regione %in% lista[[1]]$regioni) %>% summarise(soglia_iniziale = sum(posti_TI),
                                                                       soglia_attuale = sum(totale_ti))
s_centro <- ti %>% filter(regione %in% lista[[2]]$regioni) %>% summarise(soglia_iniziale = sum(posti_TI),
                                                                         soglia_attuale = sum(totale_ti))
s_sud <-ti %>% filter(regione %in% lista[[3]]$regioni) %>% summarise(soglia_iniziale = sum(posti_TI),
                                                                     soglia_attuale = sum(totale_ti))
s_isole <-ti %>% filter(regione %in% lista[[4]]$regioni) %>% summarise(soglia_iniziale = sum(posti_TI),
                                                                       soglia_attuale = sum(totale_ti))
s_nord
s_centro
s_sud
s_isole
