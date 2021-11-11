#!/usr/bin/env python3

import json
import os
from datetime import datetime
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import textstat
import pickle
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def parse_entries(filePath):
	entryDict = {}
	dateFormat = re.compile(r'\d\d_\d\d_\d\d\.json')
	for file in os.listdir(filePath):
		if dateFormat.match(os.path.basename(file)):
			with open(os.path.join(filePath, file), 'r') as f:
				entryPointDictList = json.load(f)['listContent']
				entryPointList = [point['text'] for point in entryPointDictList]
				date = datetime.strptime(os.path.splitext(file)[0], '%d_%m_%y')
				entryDict[date] = entryPointList
	sortedDates = sorted(list(entryDict.keys()))
	return entryDict, sortedDates

def get_text(entryDict):
	entryPointList = entryDict.values()
	entryList = ['\n'.join(points) for points in entryPointList]
	text = '\n'.join(entryList)
	return text

def save_figure(figureName):
	plt.savefig(f'{figureName}.png', dpi=200, bbox_inches='tight', pad_inches=0.2)
	return

def get_entry_lengths(entryDict, sortedDates):
	entryLengths = []
	for date in sortedDates:
		entryText = '\n'.join(entryDict[date])
		entryLengths.append(len(entryText))
	return entryLengths

def plot_entry_lengths(sortedDates, entryLengths):
	plt.plot(sortedDates, entryLengths)
	plt.xticks(rotation='vertical')
	plt.xlabel("Entry date")
	plt.ylabel("Entry length (characters)")
	return plt.plot

def get_instrument_frequencies(instrumentSet, entryDict, sortedDates):
	instrumentFrequencyDict = {}
	for instrument in instrumentSet:
		instrumentFrequencyList = []
		for date in sortedDates:
			entryText = '\n'.join(entryDict[date])
			entryInstrumentFrequency = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(instrument), entryText))
			instrumentFrequencyList.append(entryInstrumentFrequency)
		instrumentFrequencyDict[instrument] = np.cumsum([0] + instrumentFrequencyList[:-1]).tolist()
	return instrumentFrequencyDict

def plot_instrument_frequencies(sortedDates, instrumentFrequencyDict):
	for instrument, instrumentFrequencies in instrumentFrequencyDict.items():
		plt.plot(sortedDates, instrumentFrequencies, label=instrument)
	plt.xticks(rotation='vertical')
	plt.xlabel("Entry date")
	plt.ylabel("Instrument mentions (cumulative)")
	plt.legend()
	return plt.plot

def get_lexicon(text):
	lexicon = textstat.lexicon_count(text, removepunct=True)
	return lexicon

def get_readability(text):
	readability = textstat.flesch_reading_ease(text)
	return readability

def get_tokens(text, textName):
	print(f"Getting tokens from {textName}...") 
	tokens = nltk.sent_tokenize(text)
	#tokens = list(filter(lambda token: token not in string.punctuation, tokens))
	tokenFileName = f'{textName}_tokens' 
	with open(tokenFileName, 'wb') as tokenFile:
    		pickle.dump(tokens, tokenFile)
	return tokenFileName

def analyse_sentiment(text, textName):
	with open (f'{textName}_tokens', 'rb') as tokenFile:
		tokens = pickle.load(tokenFile)
	#stopwords = nltk.corpus.stopwords.words("english")
	#contentTokens = [token for token in tokens if token.lower() not in stopwords]
	sid = SentimentIntensityAnalyzer()
	negSentimentList = []
	posSentimentList = []
	for sentence in tokens:
		negSentimentList.append(sid.polarity_scores(sentence)['neg'])
		posSentimentList.append(sid.polarity_scores(sentence)['pos'])
	return np.mean(posSentimentList) - np.mean(negSentimentList)

if __name__ == "__main__":
	filePath = os.path.abspath('takeout-20211110T024606Z-001/Keep/')
	
	entryDict, sortedDates = parse_entries(filePath)
	text = get_text(entryDict)

	#instrumentSet = {'piano',
	#		 'accordion',
	#		 'guitar',
	#		 'bass',
	#		 'balalaika',
	#		 'sanshin',
	#		 'mandolin',
	#		 'bodhran',
	#		 'banjo',
	#		 'tin whistle',
	#		 'harmonica'}

	#entryLengths = get_entry_lengths(entryDict, sortedDates)
	#plot_entry_lengths(sortedDates, entryLengths)
	#save_figure('entry_lengths')

	#instrumentFrequencyDict = get_instrument_frequencies(instrumentSet, entryDict, sortedDates)
	#plot_instrument_frequencies(sortedDates, instrumentFrequencyDict)
	#save_figure('instrument_frequencies')
	
	lexicon = get_lexicon(text)
	print(f"lexicon = {lexicon} words")

	readability = get_readability(text)
	print(f"Flesch–Kincaid reading ease score = {readability:.2f}")
	
	textName = 'text'
	#tokenFileName = get_tokens(text, textName)
	sentiment = analyse_sentiment(text, textName)
	if sentiment > 0:
		print(f"sentiment = {sentiment} (optimist)")
	else:
		print(f"sentiment = {sentiment} (pessimist)")
	sys.exit(0)

