#!/usr/bin/env python
# coding: utf-8

# 


#For labbelling tweets into 8 categories
import pandas as pd
import csv

sevD = {}

sevD["accident"] = ["#busaccident" or "bus accident" or "vehicle accident" or "car accident" or "#CarAccident" or "bike accident" or "motorcycle" or "MotorcycleAccident" or "#BikeAccident" or "#bikeaccident" or "vehicle accident"]
sevD["business"] = ["business" or "money" or "nifty" or "sensex" or "stock market" or "industry" or "company" or "marketing" or "selling" or "profit" or "loss" or "investment" or "customers" or "product" or "enterprise"]
sevD["cricket"] = ["cricket" or "wicket" or "sachin tendulkar" or "IPL" or "BCCI" or "ICC" or "cricket world cup" or "test match" or "umpire" or "T20" or "runout" or "batsman" or "bowler" or "wicketkeeper" or "catch" or "stump"]
sevD["education"] = ["education" or "UGC" or "AICTE" or "IIT" or "college" or "NIT" or "University" or "school" or "student" or "teacher" or "library" or "coaching" or "knowledge" or "scholarship" or "MHRD" or 'book' or "seminar" or "conference"]
sevD["entertainment"] = ["bollywood" or "entertainment" or "movie" or "cinema" or "video" or "youTube" or "amusemnet"]
sevD["health"] = ["vaccine" or "coronavirus" or "corona" or "COVID-19" or "physical health" or "health" or "doctor" or "patient" or "medicine" or "meditation" or "exercise" or "yoga" or "fitness" or "illness"]
sevD["politics"] = ["politics" or "government" or "political party" or "political alliance" or "opposition party" or "election" or "parliament" or "vote" or "PM" or "CM" or "BJP" or "Narendra Modi" or "congress"]
sevD["sports"] = ["sports" or "game" or "olympics" or "athletics" or "badminton" or "football" or "Asian games" or "FIFA world cup" or "Asia Cup" or "commonwealth game" or "Wimbledon" or "Pro Kabaddi"]
def findSev(tw):
    for s,l in sevD.items():
        for w in l:
            if w in tw:
                return s
    return None
#Reading the crawled Twitter data.csv file

df = pd.read_csv('SRTwitter_Topic_Classification_dataset.csv')

tweets = df["text"].tolist()

#print(tweets)


sev_pred = []

for t in tweets:
    sev_pred.append(findSev(t))


df["label1"] = sev_pred
#Labeled Twitter dataset
df.to_csv('Labeled_SRTwitter_Topic_Classification_dataset.csv',index=False)



