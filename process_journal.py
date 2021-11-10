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
	return entryDict

def get_text(entryDict):
	entryPointList = entryDict.values()
	entryList = ['\n'.join(points) for points in entryPointList]
	text = '\n'.join(entryList)
	return text

def save_figure(figureName):
	plt.savefig('{}.png'.format(figureName), dpi=200, bbox_inches='tight', pad_inches=0.2)

def get_entry_lengths(entryDict):
	entryLengths = []
	sortedDates = sorted(list(entryDict.keys()))
	for date in sortedDates:
		entryText = '\n'.join(entryDict[date])
		entryLengths.append(len(entryText.strip()))
	return sortedDates, entryLengths

def plot_entry_lengths(sortedDates, entryLengths):
    	plt.plot(sortedDates, entryLengths)
	plt.xticks(rotation='vertical')
	plt.xlabel("Entry date")
	plt.ylabel("Entry length (characters)")
	return plt.plot

def get_instrument_frequencies(instrumentSet, entryDict):
	instrumentFrequencies = {}
	for date in entryDict.keys():
		entryInstrumentFrequencies = {}
		entryText = '\n'.join(entryDict[date])
		for instrument in instrumentSet:
			entryInstrumentFrequency = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(instrument), entryText))
			entryInstrumentFrequencies[instrument] = entryInstrumentFrequency
		instrumentFrequencies[date] = (entryInstrumentFrequencies)
	return instrumentFrequencies

def plot_instrument_frequencies(instrumentFrequencies):
	#sortedDates = sorted(instrumentFrequencies.values())
	#instrumentFrequencyList = []
	#for date in sortedDates:
	#	instrumentFrequencyList.append(np.array(instrumentFrequencies[date]))
	#instrumentFrequencyArray = np.array(instrumentFrequencyList)
	#print(instrumentFrequencyArray)
	
	#plt.bar(range(len(instrumentFrequencies)), list(instrumentFrequencies.values()), align='center')
	#plt.xticks(range(len(instrumentFrequencies)), list(instrumentFrequencies.keys()))
	return plt.plot

if __name__ == "__main__":
	filePath = os.path.abspath('takeout-20211110T024606Z-001/Keep/')
	
	entryDict = parse_entries(filePath)

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


	sortedDates, entryLengths = get_entry_lengths(entryDict)
	plot_entry_lengths(sortedDates, entryLengths)
	save_figure('entry_lengths')
	sys.exit(0)

	instrumentFrequencies = get_instrument_frequencies(instrumentSet, entryDict)
	plot_instrument_frequencies(instrumentFrequencies)
	plt.show()

