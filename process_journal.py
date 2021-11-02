#!/usr/bin/env python3

import json
import os
import datetime
import re
import sys
import matplotlib.pyplot as plt


def parse_entries(filePath):
	entryDict = {}
	dateFormat = re.compile(r'\d\d_\d\d_\d\d\.json')
	for file in os.listdir(filePath):
		if dateFormat.match(os.path.basename(file)):
			with open(os.path.join(filePath, file), 'r') as f:
				entryPointDictList = json.load(f)['listContent']
				entryPointList = [point['text'] for point in entryPointDictList]
				entryDict[os.path.splitext(file)[0]] = entryPointList
	return entryDict

def get_text(entryDict):
	entryPointList = entryDict.values()
	entryList = ['\n'.join(points) for points in entryPointList]
	text = '\n'.join(entryList)
	return text

def match_instruments(instrumentSet, text):
	instrumentMatchList = list(filter(lambda v: re.match('\w', text), instrumentSet))
	return instrumentMatchList

def get_instrument_frequencies(instrumentSet, entryDict):
	for entry in entryDict.keys():
		entryText = '\n'.join(entry)

	return instrumentFrequencies

def plot_instrument_frequencies(instrumentFrequencies):
	return plt.plot

if __name__ == "__main__":
	filePath = os.path.abspath('takeout/Keep/')
	
	entryDict = parse_entries(filePath)
	sys.exit(0)

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

	instrumentFrequencies = get_instrument_frequencies(instrumentSet, entryDict)
	plot_instrument_frequencies(instrumentFrequencies)
	plt.show()

