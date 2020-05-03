library(writexl)
library(dplyr)
library(geojsonio)
data <- read.csv("https://github.com/pcm-dpc/COVID-19/raw/master/dati-regioni/dpc-covid19-ita-regioni.csv", stringsAsFactors = FALSE)
ti <- readxl::read_xlsx("data/nuovi_posti_TI.xlsx")

data <- data %>% 
  mutate(data = as_date(data)) %>% 
  filter(data == max(data)) %>% 
  select(regione= denominazione_regione, terapia_intensiva, totale_positivi)

#merge PA
merge <- c("P.A. Bolzano", "P.A. Trento")
data[nrow(data)+1, 1] <- "Trentino-Alto Adige"
data[nrow(data), 2:ncol(data)] <-  apply(data[data$regione %in% merge,2:ncol(data)], 2, sum)
data <- data[!(data$regione %in% merge),] %>% 
  arrange(regione)

data$posti_ti <- ti$posti_TI
data$nuovi_posti_ti <- ti$nuovi_posti_TI
data$totale_ti <- data$nuovi_posti_ti + data$posti_ti
data$saturazione <- round(data$terapia_intensiva/data$totale_ti, 2)*100


names(data)[1] <- "description"

data[data$description =="Emilia Romagna", 1] <- "Emilia-Romagna"
data[data$description =="Friuli Venezia Giulia", 1] <- "Friuli-Venezia Giulia"
data <- data %>% 
  mutate(area = case_when(description %in% c("Sardegna", "Sicilia") ~ "ISOLE",
                          description %in% c("Basilicata", "Calabria","Campania","Puglia") ~ "SUD",
                          description %in% c("Abruzzo", "Campania","Lazio","Marche","Molise","Toscana","Umbria") ~ "CENTRO",
                          description %in% c("Emilia-Romagna","Friuli-Venezia Giulia","Liguria","Lombardia","Piemonte","Trentino-Alto Adige","Valle d'Aosta","Veneto") ~ "NORD"),
         colore = case_when(area == "NORD" ~ "#cc503e",
                            area == "CENTRO" ~ "#edad08",
                            area == "SUD" ~ "#1d6996",
                            area == "ISOLE" ~ "#73af48"))
map <- geojson_read("data/regioni_italiane.geojson",  what = "sp")

map@data <- map@data %>% 
  left_join(data)

geojson_write(map, file=paste0("export/3_mappa_terapiaintensiva_", format(Sys.Date(), "%d%m"),".geojson"))
