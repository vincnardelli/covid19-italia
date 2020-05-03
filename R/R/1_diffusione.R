library(dplyr)
library(writexl)
library(deSolve)
library(lubridate)
data <- read.csv("https://github.com/pcm-dpc/COVID-19/raw/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")

#R0 nazionale (Andrea)

nation <- read_csv("fit_modelli/r0.csv")
nation <- cbind(nation,"Giorni" = c(5:(nrow(nation)+4)))

nation <- nation %>% select(Data,R0) %>% pivot_wider(names_from = Data, values_from = R0)
nation <- cbind(data.frame("data"= "Un infetto contagia altri"), nation)

write.xlsx(nation, "export/r0_nation.xlsx")


#R0 regioni

r0_regions <- read_csv("fit_modelli/r0_regions.csv")
r0_regions %>% rename("Nord" = nord,"Centro" = centro, "Sud" = sud,"Isole" = isole) %>% 
  gather(Nord,Centro,Sud,Isole, key = Area, value = r0 ) %>% 
  pivot_wider(names_from = Data, values_from = r0) %>%
  write.xlsx("export/r0_regions.xlsx")














#R0 sardegna
#r0_sard <-  read_csv("fit_modelli/r0_Sardegna.csv")
#r0_sard <- r0_sard %>% filter(Data >= "2020-03-16") %>% pivot_wider(names_from = Data, values_from = sardegna)
#r0_sard <- cbind(data.frame("data"= "Un infetto contagia altri"), r0_sard)
#write.xlsx(r0_sard, "export/r0_sardegna.xlsx")
#
##R0 friuli
#r0_friuli <-  read_csv("fit_modelli/r0_friuli.csv")
#r0_friuli <- r0_friuli %>%  pivot_wider(names_from = Data, values_from = nord)
#r0_friuli <- cbind(data.frame("data"= "Un infetto contagia altri"), r0_friuli)
#write.xlsx(r0_friuli, "export/r0_friuli.xlsx")



# vecchio R0 vinc

#R0l <- c()
#for(i in 2:nrow(data)){
#  data2 <- data[1:i, ]
#  # cat(i, '\n')
#  Infected <- data2$totale_positivi
#  Day <- 1:(length(Infected))
#  N <- 60000000
#  
#  
#  SIR <- function(time, state, parameters) {
#    par <- as.list(c(state, parameters))
#    with(par, {
#      dS <- -beta/N * I * S
#      dI <- beta/N * I * S - gamma * I
#      dR <- gamma * I
#      list(c(dS, dI, dR))
#    })
#  }
#  
#  
#  init <- c(S = N-Infected[1], I = Infected[1], R = 0)
#  RSS <- function(parameters) {
#    names(parameters) <- c("beta", "gamma")
#    out <- ode(y = init, times = Day, func = SIR, parms = parameters)
#    fit <- out[ , 3]
#    sum((Infected - fit)^2)
#  }
#  
#  Opt <- optim(c(0.5, 0.5), RSS, method = "L-BFGS-B", lower = c(0, 0), upper = c(1, 1)) 
#  
#  Opt_par <- setNames(Opt$par, c("beta", "gamma"))
#  Opt_par
#  
#  R0 <- setNames(Opt_par["beta"] / Opt_par["gamma"], "R0")
#  R0l <- c(R0l, R0)
#  
#  
#}
#
#data$data <- as_date(data$data)
#
#df <- data.frame(data=format(data$data[2:nrow(data)], '%d/%m'), 
#                 R0=round(R0l, 2),
#                 giorni = c(2:nrow(data)))
#df <- df %>% rename("Giorni dall'inizio del tracciamento dati" = giorni)
#
#write_xlsx(df, paste0("export/1_diffusione_", format(Sys.Date(), "%d%m"),".xlsx"))