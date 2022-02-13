#!/bin/bash
python process_journal.py\
    -journalPath Takeout/Keep/\
    --instrumentsPath instruments.txt\
    --tinnitusKeywordsPath tinnitus_keywords.txt\
    --stopwordSupplementPath stopword_supplement.txt\
    --lexicon\
    --readability\
    --sentiment\
    --entryLengths\
    --wordcloud\
> metrics.log