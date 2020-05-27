# rischio provinciale
# Si tratta di monitorare il numero di nuove infezioni negli ultimi 7 giorni:
# se supera la soglia di 50 ogni 100mila abitanti, si frena.
library(readxl)
library(readr)
library(lubridate)
library(dplyr)
library(geojsonio)
library(writexl)
library(openxlsx)


abitanti <- read.xlsx("data/abitanti_istat.xlsx")

province <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv")
province$data <- as_date(province$data)
province <- province %>% filter(denominazione_provincia != "In fase di definizione/aggiornamento")

province$denominazione_regione <-  gsub("P.A. Bolzano", "Trentino Alto Adige", province$denominazione_regione)
province$denominazione_regione <- gsub("P.A. Trento", "Trentino Alto Adige", province$denominazione_regione)

province <- province %>% arrange(desc(data)) %>% rename("Data" = data,
                                                        "Regione" = denominazione_regione,
                                                        "Provincia" = denominazione_provincia,
                                                        "Totale casi" = totale_casi) %>% 
  select(Data, Regione, Provincia, `Totale casi`)


province <- left_join(province,abitanti)

prima <- province %>% filter(Data == Sys.Date() - 7) %>% select(Provincia,`Totale casi`) %>% rename(prima = `Totale casi`)

variazione <- left_join(province,prima) 

variazione <- variazione %>% mutate("Variazione" = `Totale casi` - prima,
                                    "Target" = round((abitanti/100000)*50,2),
                                    "Var_100k" = round(Variazione/(as.numeric(abitanti)/100000),2))
variazione  %>% filter(Data == max(Data)) %>% select(Provincia,abitanti, Variazione,Var_100k)  %>% 
  arrange(desc(Var_100k)) %>% 
  rename("Abitanti" = abitanti, "Contagi ultimi 7 giorni"= Variazione, "Contagi ultimi 7 giorni ogni 100k ab" = Var_100k) %>% 
  write.xlsx("export/tabella_rischio.xlsx")




variazione <- variazione %>% rename("prov_name" =Provincia) %>% filter(Data == max(Data))


map <- geojson_read("data/province.geojson",  what = "sp")
map@data$prov_name <- as.character(map@data$prov_name)
map@data$prov_name[map@data$prov_name == "Valle d'Aosta/Vallée d'Aoste"] <- "Aosta"
map@data$prov_name[map@data$prov_name == "Bolzano/Bozen"] <- "Bolzano"
map@data$prov_name[map@data$prov_name == "Massa-Carrara"] <- "Massa Carrara"

map@data <- map@data %>% left_join(variazione)
#map@data %>% View()


geojson_write(map, file = paste0("export/province_rischio_",format(Sys.Date(), "%d%m"),".geojson"))






