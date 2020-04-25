
# TWITTER DATA ANALYSIS ------------------------------------------------------------------------------------

# load twitter library - the rtweet library is recommended now over twitteR
library(rtweet)
library(ggplot2)
library(dplyr)
library(tidytext) #text mining library
library(twitteR)
library(rjson)
library(tm)
library(stringr)
library(wordcloud)
library(wordcloud2)
library(dplyr)
library(stringr)
library(SnowballC)
library(RColorBrewer)
library(syuzhet)
library(topicmodels)
library(textdata)

#app name
appname <- "app name"

## api key
key <- "api key"

## api secret
secret <- "api secret"


# create token named "twitter_token"
twitter_token <- create_token(
  app = appname,
  consumer_key = key,
  consumer_secret = secret)

## search for 500 tweets using the #rstats hashtag
rdc <- search_tweets(q = "#Covid_19", # and #conferenzastampa
                     n = 5000, lang="it")
i<- which( rdc$is_retweet == TRUE)
rdc.new<-rdc[i,]
which(rdc$text != rdc$retweet_text)
rdc.table <- as.data.frame(rdc)
tweet<-rdc$hashtags
tweet<-unlist(tweet)
rdc.table <- apply(rdc.table,2,as.character)
write.csv(rdc.table,"covid_10_04.csv")
rdc.table <- read.csv("covid_10_04.csv")
rdc.table <- apply(rdc.table,2,as.character)
rdc <- read.csv("covid_10_04.csv")
a<-vector()
d<-as.data.frame(rdc,stringsAsFactors=FALSE)

df <- read.csv("covid_10_04.csv", sep=",", header = T, stringsAsFactors=FALSE)[-1]

# Cleaning and preliminary analysis ----

df$text <- sapply(df$text,function(row) iconv(row, "latin1", "ASCII", sub="")) #remove special characters
myCorpus <- Corpus(VectorSource(df$text)) #create the Text Mining object
myCorpus <- tm_map(myCorpus, content_transformer(tolower)) #everything in lower case
removeURL <- function(x) gsub("http[^[:space:]]*", "", x) 
myCorpus <- tm_map(myCorpus, content_transformer(removeURL)) #remove URL
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x) 
myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct)) #remove punctuation
myCorpus <- tm_map(myCorpus, removeNumbers) #remove numbers
myStopwords <- c(stopwords(kind="it"), "fufuuaefufuub", "fufuuafufuuaa", "fufuuf", 
                 "fufuuaafufuubacovid", "fufuuallo" ,   "fufuub", "uufef", "via", 
                 "perch", "due", "fufuu", "fufuuu", "quando", "perch", "cos", "finch", "fufuuac",
                 "uaaufef","ucucuc", "sar", "fufuuffufuuffufuuf", "cazzo") 
myCorpus <- tm_map(myCorpus, removeWords, myStopwords) #remove stopwords

#function that counts words
tdm <- TermDocumentMatrix(myCorpus, control=list(wordLengths=c(1,Inf)))
tdm <- TermDocumentMatrix(myCorpus, control=list(minWordLength=1))

tdm1 = TermDocumentMatrix(myCorpus,
                          control = list(weighting = weightTfIdf))

# Wordcloud ----
m <- as.matrix(tdm1)
wordFreq <- sort(rowSums(m), decreasing=TRUE)
redPalette <- c("#5c1010", "#6f0000", "#560d0d", "#c30101", "#940000")
dev.new()
wordcloud(words=names(wordFreq), freq=wordFreq, min.freq=35, random.order=F, colors=redPalette)

# Cluster ----

#function for word clustering: 
tdm2 <- removeSparseTerms(tdm1, sparse=0.99)
m2 <- as.matrix(tdm2)
distMatrix <- dist(scale(m2))
fit <- hclust(distMatrix, method="ward.D")
dev.new()
plot(fit, labels=F)
rect.hclust(fit, k=5)
groups <- cutree(fit, k=5)
table(groups) #in the second group there's only climatechange

# define dendrogram object to play with:
hc <- fit
dend <- as.dendrogram(hc)

library(dendextend)
par(mfrow = c(1,1), mar = c(5,2,1,0))
dend <- dend %>%
  color_branches(k = 7) %>%
  set("branches_lwd", c(1,2,1)) %>%
  set("branches_lty", c(2,1,2)) %>%
  set("labels_cex", 0.65) 

dend <- color_labels(dend, k = 5)
dev.new()
plot(dend, horiz=T)

# LDA (Latent Dirichlet Allocation) ------------------------------------------------------------------------

library(topicmodels)

#Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

#Create document-term matrix
dtm1 <- as.DocumentTermMatrix(tdm2, control=list(weighting=identity))
dtm1 <- weightTf(dtm1)
rownames(dtm1) <- df$text
dtm1$v <- rep.int(1, 44056) # (1, length of i in tdm2)
rowTotals <- apply(dtm1, 1, sum) #Find the sum of words in each Document
dtm1   <- dtm1[rowTotals> 0, ]  
m1 <- as.matrix(dtm1)

#Number of topic:

#install.packages("ldatuning")
library("ldatuning")

result <- FindTopicsNumber(
  dtm1,
  topics = seq(from = 2, to = 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin, alpha=1/169),
  mc.cores = 2L,
  verbose = TRUE
); result
# alpha equal to 1/nrow in tdm2

FindTopicsNumber_plot(result[,-2]) #Griffiths2004 doesn't converg

#6
result[which(result[,5] == max(result[,5])),1] #6
result[which(result[,3] == min(sort(result[6:19,3]))),1] #6

#To find the best number of cluster:
library(doParallel)
library(ggplot2)
library(scales)

#Run LDA using Gibbs sampling:

#Number of topics
k <- 6
ldaOut <-LDA(dtm1, k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, 
                                                   burnin = burnin, iter = iter, thin=thin, 
                                                   verbose = 1, alpha=1/169, delta=1))

#write out results

chapter_topics <- tidy(ldaOut, matrix = "beta")
top_terms <- chapter_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
top_terms <- top_terms[-c(1,11,19),] # remove extra words
fix(top_terms) # add à,é,è,ò,ì
top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

#beta = frequency of the word in topics
#gamma = how much the tweet contain that topic 


library(reshape2)
#comparison wordcloud with topic
dev.new()
top_terms %>%
  mutate(topic = paste("topic", topic)) %>%
  acast(term ~ topic, value.var = "beta", fill = 0) %>%
  comparison.cloud(scale=c(2,2),colors = c("brown1", 
                                           "yellowgreen", 
                                           "springgreen4", 
                                           #"deepskyblue1",
                                           "dodgerblue3", 
                                           "violet"),
                   max.words = 300, title.size=1.1)
