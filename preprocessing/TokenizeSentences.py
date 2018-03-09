from nltk.tokenize import WordPunctTokenizer
def TokenizeSentences(data):
	dataT = []
	for i in range(len(data)):
		dataT.append(WordPunctTokenizer().tokenize(data[i]))
	return dataT