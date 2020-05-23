# Config file
open_browser <- FALSE

steps <- list(
  #1 
  list(path = "R/1_diffusione.R", 
       webpage = c("https://app.flourish.studio/visualisation/1883224/edit",
                   "https://app.flourish.studio/visualisation/1889832/edit")), 
  #2
  list(path = "R/2_efficacia_misure.R", 
       webpage = c("https://app.flourish.studio/visualisation/1793321/edit")),
  #3
  list(path = "R/3_mappa.R", 
       webpage = c("https://app.flourish.studio/visualisation/1793308/edit")),
  #4
  list(path = "R/4_ti_regioni.R", 
       webpage = c("https://app.flourish.studio/visualisation/1810005/edit",
                   "https://app.flourish.studio/visualisation/1810311/edit",
                   "https://app.flourish.studio/visualisation/1810329/edit",
                   "https://app.flourish.studio/visualisation/1810346/edit",
                   "https://app.flourish.studio/story/247628/edit")),
  #5
  list(path = "R/5_sir_nazionale.R", 
       webpage = c("https://app.flourish.studio/visualisation/1793293/edit",
                   "https://app.flourish.studio/visualisation/1793303/edit",
                   "https://app.flourish.studio/visualisation/1997920/edit",
                   "https://app.flourish.studio/visualisation/2141336/edit")),
  #6
  list(path = "R/6_regioni_race.R", 
       webpage = c("https://app.flourish.studio/visualisation/1793324/edit")),
  #7
  list(path = "R/7_sir_regioni.R", 
       webpage = c("https://app.flourish.studio/visualisation/1801654/edit",
                   "https://app.flourish.studio/visualisation/1801658/edit",
                   "https://app.flourish.studio/visualisation/1801663/edit",
                   "https://app.flourish.studio/visualisation/1801665/edit",
                   "https://app.flourish.studio/story/246458/edit")),
  #8
  list(path = "R/8_dati_ufficiali.R",
       webpage = c("https://app.flourish.studio/visualisation/1794060/edit",
                   "https://app.flourish.studio/visualisation/1813414/edit",
                   "https://app.flourish.studio/visualisation/1813557/edit")),
  #9
  list(path = "R/9_tamponi.R"),
  
  #10
  list(path = "R/10_internazionale.R",
       webpage = c("https://app.flourish.studio/visualisation/2108463/edit",
                   "https://app.flourish.studio/visualisation/2078503/edit",
                   "https://app.flourish.studio/visualisation/2074767/edit",
                   "https://app.flourish.studio/visualisation/2137348/edit",
                   "https://app.flourish.studio/visualisation/2112167/edit",
                   "https://app.flourish.studio/visualisation/2112323/edit",
                   "https://app.flourish.studio/visualisation/2112402/edit",
                   "https://app.flourish.studio/story/293444/edit",
                   "https://app.flourish.studio/visualisation/2129608/edit")),
  #11
  list(path = "R/11_provinciale.R")
  
)



# Functions

clean <- function(active=TRUE){
  l <- list.files("export", full.names = TRUE)
  if(active & length(l) > 0){
    do.call(file.remove, list(l))
  }
}


step <- function(id){
  cat("Running step", id,  "...\n")
  source(steps[[id]]$path)
  cat("done ✔ \n")
  
  if(length(steps[[id]]$webpage) >0 & open_browser){
    for(i in 1:length(steps[[id]]$webpage)){
      browseURL(steps[[id]]$webpage[i], browser = getOption("browser"), encodeIfNeeded = FALSE)
      
    }
  }
  
}
