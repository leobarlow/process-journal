#!/usr/bin/env python3

import json
import os
from datetime import datetime
import re
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from wordcloud import WordCloud
import argparse
import collections


parser = argparse.ArgumentParser()
parser.add_argument('-journalPath', dest='journalPath', action='store', type=str, required=True, help="Path for Takeout data")
parser.add_argument('--instrumentsPath', dest='instrumentsPath', action='store', type=str, required=False, help="Path for list of instruments")
parser.add_argument('--tinnitusKeywordsPath', dest='tinnitusKeywordsPath', action='store', type=str, required=False, help="Path for list of tinnitus keywords")
parser.add_argument('--stopwordSupplementPath', dest='stopwordSupplementPath', action='store', type=str, required=False, help="Path for stopword supplement list")
parser.add_argument('--timestamps', default=False, required=False, action='store_true', help="Plot timestamps over time")
parser.add_argument('--sentiment', default=False, required=False, action='store_true', help="Print mean sentence sentiment")
parser.add_argument('--entryLengths', default=False, required=False, action='store_true', help="Plot entry lengths over time")
parser.add_argument('--wordcloud', default=False, required=False, action='store_true', help="Generate content word wordcloud")
parser.add_argument('--nameFrequencies', default=False, required=False, action='store_true', help="Plot mentioned names by frequency")
parser.add_argument('--wordcount', default=False, required=False, action='store_true', help="Print wordcount")


def parse_entries(journalPath, dateFormat):
	entryDict = {}
	timestampDict = {}
	for file in os.listdir(journalPath):
		if dateFormat.match(os.path.basename(file)):
			with open(os.path.join(journalPath, file), 'r') as f:
				entryPointDictList = json.load(f)['listContent']
				entryPointList = [point['text'] for point in entryPointDictList]
				date = datetime.strptime(os.path.splitext(file)[0], '%d_%m_%y')
				entryDict[date] = entryPointList
			with open(os.path.join(journalPath, file), 'r') as f:
				date = datetime.strptime(os.path.splitext(file)[0], '%d_%m_%y')
				ts = datetime.utcfromtimestamp(json.load(f)['userEditedTimestampUsec']/1000000)
				if ts.hour < 12:
					ts = ts.replace(year=1, month=1, day=2)
				else:
					ts = ts.replace(year=1, month=1, day=1)
				timestampDict[date] = ts
	sortedDates = sorted(list(entryDict.keys()))
	return entryDict, timestampDict, sortedDates

def parse_timestamps(journalPath, dateFormat):
	timestampDict = {}
	for file in os.listdir(journalPath):
		if dateFormat.match(os.path.basename(file)):
			with open(os.path.join(journalPath, file), 'r') as f:
				date = datetime.strptime(os.path.splitext(file)[0], '%d_%m_%y')
				ts = datetime.utcfromtimestamp(json.load(f)['userEditedTimestampUsec']/1000000)
				if ts.hour < 12:
					ts = ts.replace(year=1, month=1, day=2)
				else:
					ts = ts.replace(year=1, month=1, day=1)
				timestampDict[date] = ts
	return timestampDict

def get_text(entryDict):
	entryPointList = entryDict.values()
	entryList = [' '.join(points) for points in entryPointList]
	text = ' '.join(entryList)
	return text

def save_figure(figureName):
	plt.savefig(f'figs/{figureName}.png', dpi=200, bbox_inches='tight', pad_inches=0.2)
	plt.cla()
	return

def get_entry_lengths(entryDict, sortedDates):
	entryLengths = []
	for date in sortedDates:
		entryText = '\n'.join(entryDict[date])
		entryLengths.append(len(entryText))
	return entryLengths

def get_timestamps(timestampDict, sortedDates):
	timestamps = []
	for date in sortedDates:
		timestamps.append(timestampDict[date])
	return timestamps

def get_moving_average(interval, window):
    window = np.ones(int(window))/float(window)
    return np.convolve(interval, window, 'same')

def plot_entry_lengths(sortedDates, entryLengths):
	plt.plot(sortedDates, entryLengths)
	y_av = get_moving_average(entryLengths, window=100)
	plt.plot(sortedDates, y_av)
	plt.xticks(rotation='vertical')
	plt.xlabel("Entry date")
	plt.ylabel("Entry length (characters)")
	return plt.plot

def plot_timestamps(sortedDates, timestamps):
	plt.scatter(sortedDates, timestamps, s=5)
	yformatter = mdates.DateFormatter('%H:%M')
	plt.gcf().axes[0].yaxis.set_major_formatter(yformatter)
	plt.xticks(rotation='vertical')
	plt.xlabel("Entry date")
	plt.ylabel("Timestamp")
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

def get_tokens(text, textName, level='word'):
	# print(f"Getting {level}-level tokens from {textName}...")
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
	with open('tmp/' + tokenFileName, 'wb') as tokenFile:
		pickle.dump(contentTokens, tokenFile)
	return tokenFileName

def analyse_sentiment(text, textName):
	with open(f'tmp/{textName}_sentence_tokens', 'rb') as tokenFile:
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
	with open (f'tmp/{textName}_word_tokens', 'rb') as tokenFile:
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

def get_name_frequencies(text, number):
	names = []
	for sent in nltk.sent_tokenize(text):
		for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
			if hasattr(chunk, 'label'):
				names.append(' '.join(c[0] for c in chunk.leaves()))
	nameFrequencies = dict(collections.Counter(names).most_common(number))
	return nameFrequencies

def plot_name_frequencies(nameFrequencies):
	plt.bar(nameFrequencies.keys(), nameFrequencies.values())
	plt.xticks(rotation='vertical')
	plt.xlabel("Name")
	plt.ylabel("Frequency")
	return plt.plot

def get_wordcount(text):
	tokenizer = RegexpTokenizer(r'\w+')
	wordcount = len(tokenizer.tokenize(text))
	return wordcount

if __name__ == "__main__":
	args = parser.parse_args()

	dateFormat = re.compile(r'\d\d_\d\d_\d\d\.json')
	entryDict, timestampDict, sortedDates = parse_entries(os.path.abspath(args.journalPath), dateFormat)
	text = get_text(entryDict)
	textName = 'text'

	if args.nameFrequencies:
		nameFrequencies = get_name_frequencies(text, number=20)
		plot_name_frequencies(nameFrequencies)
		save_figure('name_frequencies')

	if args.timestamps:
		timestampDict = parse_timestamps(args.journalPath, dateFormat)
		timestamps = get_timestamps(timestampDict, sortedDates)
		plot_timestamps(sortedDates, timestamps)
		save_figure('timestamps')

	with open('tmp/' + textName + '_entries', 'wb') as entriesFile:
		pickle.dump(list(entryDict.values()), entriesFile)

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
		save_figure('tinnitus_mentions')

	if args.wordcloud:
		tokenFileName = get_tokens(text, textName)
		wordcloud = generate_wordcloud(textName)
		plot_wordcloud(wordcloud)
		save_figure('wordcloud')

	if args.wordcount:
		wordcount = get_wordcount(text)
		print(f"wordcount = {wordcount} words ({round(wordcount/97236, 1)} Harry Potter books!)")
