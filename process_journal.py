	#!/usr/bin/env python3

import json
import os
from datetime import datetime
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pprint

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
	plt.savefig('{}.png'.format(figureName), dpi=200, bbox_inches='tight', pad_inches=0.2)
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

if __name__ == "__main__":
	filePath = os.path.abspath('takeout-20211110T024606Z-001/Keep/')
	
	entryDict, sortedDates = parse_entries(filePath)

	instrumentSet = {'piano',
			 'accordion',
			 'guitar',
			 'bass',
			 'balalaika',
			 'sanshin',
			 'mandolin',
			 'bodhran',
			 'banjo',
			 'tin whistle',
			 'harmonica'}

	#entryLengths = get_entry_lengths(entryDict, sortedDates)
	#plot_entry_lengths(sortedDates, entryLengths)
	#save_figure('entry_lengths')

	instrumentFrequencyDict = get_instrument_frequencies(instrumentSet, entryDict, sortedDates)
	plot_instrument_frequencies(sortedDates, instrumentFrequencyDict)
	save_figure('instrument_frequencies')
	sys.exit(0)

