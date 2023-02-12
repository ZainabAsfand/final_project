"""
Load scraped product items into DataFrame, write review urls into N text files.

Run example:
    $ python split_review_urls.py --scraped-products $(pwd)/../output/products_.jl --output-dir $(pwd)/../output
    $ python split_review_urls.py --scraped-products E:\Zainab\steam-scraper\output\products_all.jl --output-dir E:\Zainab\steam-scraper\output
"""
import argparse
import json
import math
import os
from random import shuffle
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import numpy as np
import pandas as pd
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scraped-products',
        help='E:\Zainab\steam-scraper\output\bug_report.csv',
    )
    parser.add_argument(
        '--output-dir',
        help='E:\Zainab\steam-scraper\output\outputs'
    )
    parser.add_argument(
        '--pieces',
        help=1,
        default=1
    )
    return parser.parse_args()

def pre_processing_techniques(sentence: str):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    nltk_tokens = nltk.word_tokenize(sentence)
    filtered_sentence = []
    for w in nltk_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

def main():
    args = parse_args()
    print(args.scraped_products)
    rows = None
    df = None
    
    df = pd.read_csv(args.scraped_products)
    blx_nontrivial = np.all(df[['title']].notnull(), axis=1)
    blx = blx_nontrivial# & blx_has_reviews
    urls = df.loc[blx, 'title'].unique()
    n = len(urls)
    fileename = args.output_dir + "review_urls_227.txt"
    print(fileename)
    with open(fileename, "w", encoding="utf-8") as file:
        for row in urls:
            line = "".join([str(x) for x in row])
            final = pre_processing_techniques(line)
            final = " ".join([str(x) for x in final])
            file.write(final + "\n")

if __name__ == "__main__":
    main()
    