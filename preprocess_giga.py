from os import walk, linesep
from os.path import join, isfile
#from sys import argv
from nltk.tokenize import word_tokenize
from collections import Counter
from operator import itemgetter

import tensorflow as tf
import numpy as np

import codecs
import json
import re

class PreProcessing:
	def file_count(self):
		''' Counts the number of files in the data path. I subtract by one to as not to count 
			the /.DS_Store file'''
		return sum([len(files) for r,d,files in walk(self.data_path)])-len(d)

	def parse_each_folder(self):
		try:
			fout = open(join(self.processed_path, '{base}-{n}'.format(base=self.processed_path, n=self.article_ndx)), 'a+')
			for root, dirs, files in walk(self.data_path):
				for name in files:
					path = join(root, name)
					if isfile(path):
						print('Currently on the file at ' + str(path))
						with codecs.open(path, 'rb', 'ascii', 'ignore') as f:
							try:
								lines = f.readlines()
								self.build_vocab(lines)

								for title, sentence in zip(self.titles, self.sentences):
									resp = self.tformat['article_pref']
									self.publisher = name[:3].upper()
									resp += self.build_postprocessed_file(sentence, title)
									self.article_ndx += 1
									if self.article_ndx != 0 and self.article_ndx % self.batch_size == 0:
										fout = open(join(self.processed_path, '{base}-{n}'.format(base=self.processed_path, n=self.article_ndx)), 'a+')
									fout.write( ("{}").format( resp.encode('utf-8', "ignore") ) )

								self.sentences = []
								self.titles = []

								if self.article_ndx % len(lines) == 0:
									self.file_ndx += 1

								 #for sent in self.sentences:
								 #	resp += tformat['sent_pref'] + sent.replace("=", "equals") + tformat['post']
								 #resp += ((tformat["post_val"] + '\t' + tformat["abs_pref"] + tformat["sent_pref"] + (lines[0]).strip('\n').replace("=", "equals") + tformat["post"] + tformat["post_val"]))
								 #data_out = open(join(self.processed_path, '{base}-{n}'.format(base=self.base_name, n=self.file_ndx)), 'a+')

							except RuntimeError as e:
									print "Runtime Error: {0} : {1}".format(e.errno, e.strerror)
		finally:
			fout.close()

	def build_vocab(self, lines):
		title = False
		article = False

		sentence_counter = 0

		curr_sentence = ''

		for line in lines:
			line = str(line)

			if '<HEADLINE>' in line:
				title = True
				sentence_counter = 0
			elif '</HEADLINE>' in line:
				title = False

			if '<P>' in line and sentence_counter == 0:
				article = True
			elif '</P>' in line and article:
				print('Article: ' + curr_sentence)
				article = False
				sentence_counter = 1
				self.sentences.append(curr_sentence[:])
				curr_sentence = ''
				self.json_elem(word_count)
			
			elif title and '<headline>' not in line.lower():
				self.article_ndx += 1
				doc_headline = line.lower()
				self.titles.append(doc_headline.split('\n')[0][:])
				print ('Title: ' + doc_headline)
				print ('Article number: ' + str(self.article_ndx))
			elif article:
				curr_line = line.lower()
				curr_sentence += ' ' + curr_line.split('\n')[0][:]
				words = word_tokenize(curr_line)
				word_count = Counter(words)
				print('Word count: ' + str(len(word_count)))
				print('Number of sentences: ' + str(sentence_counter))

	def json_elem(self, word_count):
		for key, count in word_count.items():
			if key == "":
				return
			if key not in self.vocab:
				self.vocab[key] = count
			else: self.vocab[key] += count
		print(len(self.vocab))

	def build_postprocessed_file(self, sentence, title):
		resp = (self.tformat['sent_pref'] + sentence.replace("=", "equals") + self.tformat['sent_suf'] + \
		self.tformat['abs_suf'] + '\t' + self.tformat['abs_pref'] + \
		self.tformat['sent_pref'] + title.replace('=', 'equals') + self.tformat['sent_suf'] + \
		self.tformat['sent_suf'] + \
		'\t' + 'publisher=' + self.publisher + linesep)
		return resp

	def output_vocab(self, vocab_path):
		vocabSorted = sorted(self.vocab.items(), key=itemgetter(1), reverse=True)
		with open(vocab_path, 'w+') as v:
			for item in vocabSorted:
				print >>v, ("{} {}".format(item[0], str(item[1])) )

	def vocab(self):
		return self.vocab

	def train_path(self):
		return self.train_path

	def test_path(self):
		return self.test_path

	def __init__(self, processedPath, format, fileNdx=0, batchSize = 50000):
		self.data_path = '../data_for_training'
		self.processed_path = processedPath
		self.train_path = 'train'
		self.test_path = 'test'
		self.tformat = format
		self.file_ndx = fileNdx
		self.article_ndx = 0
		self.publisher = None
		self.sentences = []
		self.titles = []
		self.batch_size = batchSize
		self.vocab = {"<UNK>":1, "<s>":1, "</s>":1, "<PAD>":1,"<d>":1,"</d>":1,"<p>":1,"</p>":1}

if __name__ == '__main__':
	processor = PreProcessing(processedPath='train', format=json.loads('{ "article_pref":"article=<d> <p> ","abs_pref":"abstract=<d> <p> ","abs_suf":" </p> </d> ","sent_pref":"<s> ", "sent_suf":" </s> "}'))
	try:
		count = processor.file_count()
		while (processor.file_ndx < count):
			processor.parse_each_folder()
		print "Each file has been parsed!"
		processor.output_vocab('data/vocab_test')
		print "The vocab has been stored!"
	except RuntimeError as e:
		print "Runtime Error: {0} : {1}".format(e.errno, e.strerror)