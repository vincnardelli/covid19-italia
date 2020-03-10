library(earlyR)
library(dplyr)
library(ggplot2)
library(deSolve)

data <- read.csv("https://github.com/pcm-dpc/COVID-19/raw/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")

Infected <- data$totale_attualmente_positivi
Day <- 1:(length(Infected))
N <- 60000000 # population of italy


SIR <- function(time, state, parameters) {
  par <- as.list(c(state, parameters))
  with(par, {
    dS <- -beta/N * I * S
    dI <- beta/N * I * S - gamma * I
    dR <- gamma * I
    list(c(dS, dI, dR))
  })
}

init <- c(S = N-Infected[1], I = Infected[1], R = 0)
RSS <- function(parameters) {
  names(parameters) <- c("beta", "gamma")
  out <- ode(y = init, times = Day, func = SIR, parms = parameters)
  fit <- out[ , 3]
  sum((Infected - fit)^2)
}

Opt <- optim(c(0.5, 0.5), RSS, method = "L-BFGS-B", lower = c(0, 0), upper = c(1, 1)) 
Opt$message

Opt_par <- setNames(Opt$par, c("beta", "gamma"))
Opt_par

t <- 1:100 
fit <- data.frame(ode(y = init, times = t, func = SIR, parms = Opt_par))
col <- 1:3 

matplot(fit$time, fit[ , 2:4], type = "l", xlab = "Day", ylab = "Number of subjects", lwd = 2, lty = 1, col = col)
matplot(fit$time, fit[ , 2:4], type = "l", xlab = "Day", ylab = "Number of subjects", lwd = 2, lty = 1, col = col, log = "y")
points(Day, Infected)
legend("bottomright", c("Susceptibles", "Infecteds", "Recovereds"), lty = 1, lwd = 2, col = col, inset = 0.05)
title("SIR model 2019-nCoV Italia", outer = TRUE, line = -2)

R0 <- setNames(Opt_par["beta"] / Opt_par["gamma"], "R0")

fit[fit$I == max(fit$I), "I", drop = FALSE] # height of pandemic

date("2020-02-24") + which(fit$I == max(fit$I))

l <- length(data$totale_attualmente_positivi)

fit$data <- seq(from=as_date(data$data[1]), to=as_date(data$data[1])+99, "days")
fit$real <- c(data$totale_attualmente_positivi, rep("", times=100-l))