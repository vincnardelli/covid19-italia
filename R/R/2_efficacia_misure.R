library(writexl)
library(lubridate)
library(tidyr)

data <- read.csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")

pos_limited <- data$totale_positivi[1:15] #positivi fino al 8 marzo
time_limited<-seq(1,15,1)
ln<-log(pos_limited)

model<-lm(ln~time_limited)

logaritmo_totale_positivi<-log(data$totale_positivi)
time <- 1:length(data$totale_positivi)
andamento_prima<-model$coefficient[1]+model$coefficient[2]*time

data$data <- as_date(data$data)

df = data.frame(matrix(vector(), 0, nrow(data)),
                 stringsAsFactors=F)
 
names(df) <- format(data$data[1:nrow(data)], '%d/%m')
df[1,] <- round(logaritmo_totale_positivi, 2)
df[2,] <- round(andamento_prima, 2)

df <- cbind(data.frame("data"= c("Con intervento", "Senza intervento")), df)

write_xlsx(df, paste0("export/2_efficaciamisure_", format(Sys.Date(), "%d%m"),".xlsx"))
