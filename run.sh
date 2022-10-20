#!/bin/bash
python process_journal.py\
    -journalPath Takeout/Keep/\
    --instrumentsPath instruments.txt\
    --tinnitusKeywordsPath tinnitus_keywords.txt\
    --stopwordSupplementPath stopword_supplement.txt\
    --timestamps\
    --sentiment\
    --entryLengths\
    --wordcloud\
    --nameFrequencies\
    --wordcount\
> tmp/metrics.log