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
import matplotlib.dates as mdates
from nltk import ngrams
from wordcloud import WordCloud
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-journalPath', dest='journalPath', action='store', type=str, required=True, help="Path for Takeout data")
parser.add_argument('--instrumentsPath', dest='instrumentsPath', action='store', type=str, required=False, help="Path for list of instruments")
parser.add_argument('--tinnitusKeywordsPath', dest='tinnitusKeywordsPath', action='store', type=str, required=False, help="Path for list of tinnitus keywords")
parser.add_argument('--stopwordSupplementPath', dest='stopwordSupplementPath', action='store', type=str, required=False, help="Path for stopword supplement list")
parser.add_argument('--lexicon', default=False, required=False, action='store_true', help="Print lexicon size")
parser.add_argument('--readability', default=False, required=False, action='store_true', help="Print readability metric")
parser.add_argument('--sentiment', default=False, required=False, action='store_true', help="Print mean sentence sentiment")
parser.add_argument('--entryLengths', default=False, required=False, action='store_true', help="Plot entry lengths over time")
parser.add_argument('--wordcloud', default=False, required=False, action='store_true', help="Generate content word wordcloud")


def parse_entries(journalPath):
	entryDict = {}
	dateFormat = re.compile(r'\d\d_\d\d_\d\d\.json')
	for file in os.listdir(journalPath):
		if dateFormat.match(os.path.basename(file)):
			with open(os.path.join(journalPath, file), 'r') as f:
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
	plt.cla()
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

def get_instrument_frequencies(instruments, entryDict, sortedDates):
	instrumentFrequencyDict = {}
	for instrument in instruments:
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

def get_tokens(text, textName, level='word'):
	print(f"Getting {level}-level tokens from {textName}...")
	if level == 'sentence':
		tokens = nltk.sent_tokenize(text)
	else:
		tokens = nltk.word_tokenize(text)
	tokens = list(filter(lambda token: token not in string.punctuation, tokens))
	stopwords = nltk.corpus.stopwords.words("english")
	if args.stopwordSupplementPath:
		with open(os.path.abspath(args.stopwordSupplementPath), 'r') as f:
			stopwords += [line.strip() for line in f]
	contentTokens = [token for token in tokens if token.lower() not in stopwords]
	tokenFileName = f'{textName}_{level}_tokens'
	with open(tokenFileName, 'wb') as tokenFile:
    		pickle.dump(contentTokens, tokenFile)
	return tokenFileName

def analyse_sentiment(text, textName):
	with open (f'{textName}_sentence_tokens', 'rb') as tokenFile:
		tokens = pickle.load(tokenFile)
	sid = SentimentIntensityAnalyzer()
	negSentimentList = []
	posSentimentList = []
	for sentence in tokens:
		negSentimentList.append(sid.polarity_scores(sentence)['neg'])
		posSentimentList.append(sid.polarity_scores(sentence)['pos'])
	return np.mean(posSentimentList) - np.mean(negSentimentList)

def get_tinnitus_mentions(tinnitusKeywords, entryDict, sortedDates):
	tinnitusMentionList = []
	entryMentionBool = False
	for date in sortedDates:
		for keyword in tinnitusKeywords:
			entryText = '\n'.join(entryDict[date])
			if sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(keyword), entryText)):
				entryMentionBool = True
		tinnitusMentionList.append(entryMentionBool)
		entryMentionBool = False
	return tinnitusMentionList

def plot_tinnitus_mentions(tinnitusMentionList, tinnitusKeywords, sortedDates):
	tinnitusMentionList = [float('nan') if x==False else x for x in tinnitusMentionList]
	plt.scatter(sortedDates, tinnitusMentionList, marker='|')
	plt.xticks(rotation='vertical')
	plt.yticks([])
	plt.xlabel("Entry date")
	plt.ylabel("Tinnitus mentions")
	plt.text(mdates.date2num(sortedDates[10]), 1.045, f"Keywords {tinnitusKeywords}", bbox=dict(alpha=0.5))
	return plt.plot

def generate_wordcloud(textName):
	with open (f'{textName}_word_tokens', 'rb') as tokenFile:
		tokens = pickle.load(tokenFile)
	tokens = ' '.join(tokens)+' '
	wordcloud = WordCloud(width = 800, height = 800,
			background_color ='white',
			min_font_size = 10).generate(tokens)
	return wordcloud

def plot_wordcloud(wordcloud):
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	return plt.plot

if __name__ == "__main__":
	args = parser.parse_args()

	entryDict, sortedDates = parse_entries(os.path.abspath(args.journalPath))
	text = get_text(entryDict)
	textName = 'text'

	if args.entryLengths:
		entryLengths = get_entry_lengths(entryDict, sortedDates)
		plot_entry_lengths(sortedDates, entryLengths)
		save_figure('entry_lengths')

	if args.instrumentsPath:
		with open(os.path.abspath(args.instrumentsPath), 'r') as f:
			instruments = set([line.strip() for line in f])
		instrumentFrequencyDict = get_instrument_frequencies(instruments, entryDict, sortedDates)
		plot_instrument_frequencies(sortedDates, instrumentFrequencyDict)
		save_figure('instrument_frequencies')

	if args.lexicon:
		lexicon = get_lexicon(text)
		print(f"lexicon = {lexicon} words")

	if args.readability:
		readability = get_readability(text)
		print(f"Flesch–Kincaid reading ease score = {readability:.2f}")

	if args.sentiment:
		tokenFileName = get_tokens(text, textName, level='sentence')
		sentiment = analyse_sentiment(text, textName)
		if sentiment > 0:
			print(f"sentiment = {sentiment} (optimist)")
		else:
			print(f"sentiment = {sentiment} (pessimist)")

	if args.tinnitusKeywordsPath:
		with open(os.path.abspath(args.tinnitusKeywordsPath), 'r') as f:
			tinnitusKeywords = set([line.strip() for line in f])
		tinnitusMentionList = get_tinnitus_mentions(tinnitusKeywords, entryDict, sortedDates)
		plot_tinnitus_mentions(tinnitusMentionList, tinnitusKeywords, sortedDates)
		save_figure('tinnitus mentions')

	if args.wordcloud:
		tokenFileName = get_tokens(text, textName)
		wordcloud = generate_wordcloud(textName)
		plot_wordcloud(wordcloud)
		save_figure('wordcloud')

	sys.exit(0)
