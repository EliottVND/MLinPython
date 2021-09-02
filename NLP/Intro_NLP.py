# Using regex 

	# Write a pattern to match sentence endings: sentence_endings
	sentence_endings = r"[.?!]"

	# Split my_string on sentence endings and print the result
	print(re.split(sentence_endings, my_string))

	# Find all capitalized words in my_string and print the result
	capitalized_words = r"[A-Z]\w+"
	print(re.findall(capitalized_words, my_string))

	# Split my_string on spaces and print the result
	spaces = r"\s+"
	print(re.split(spaces, my_string))

	# Find all digits in my_string and print the result
	digits = r"\d+"
	print(re.findall(digits, my_string))

# Using nltk to tokenize sentences and finding words and unique tokens

	# Import necessary modules
	from nltk.tokenize import sent_tokenize
	from nltk.tokenize import word_tokenize

	# Split scene_one into sentences: sentences
	sentences = sent_tokenize(scene_one)

	# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
	tokenized_sent = word_tokenize(sentences[3])

	# Make a set of unique tokens in the entire scene: unique_tokens
	unique_tokens = set(word_tokenize(scene_one))

	# Print the unique tokens result
	print(unique_tokens)


# Using search method

	# Search for the first occurrence of "coconuts" in scene_one: match
	match = re.search("coconuts", scene_one)

	# Print the start and end indexes of match
	print(match.start(), match.end())


	# Write a regular expression to search for anything in square brackets: pattern1
	pattern1 = r"\[.*]"

	# Use re.search to find the first text in square brackets
	print(re.search(pattern1, scene_one))


	# Find the script notation at the beginning of the fourth sentence and print it
	pattern2 = r"[A-Z]+:"
	print(re.match(pattern2, sentences[3]))

# Using regex with nltk

	# Import the necessary modules
	from nltk.tokenize import regexp_tokenize
	from nltk.tokenize import TweetTokenizer
	# Write a pattern that matches both mentions (@) and hashtags
	pattern2 = r"([@#]\w+)"
	# Use the pattern on the last tweet in the tweets list
	mentions_hashtags = regexp_tokenize(tweets[-1], pattern2)
	print(mentions_hashtags)

# Using TweetTokenizer

	# Import the necessary modules
	from nltk.tokenize import TweetTokenizer
	# Use the TweetTokenizer to tokenize all tweets into one list
	tknzr = TweetTokenizer()
	all_tokens = [tknzr.tokenize(t) for t in tweets]
	print(all_tokens)



# Using regex on german (for new letters) and emojiiiiis

	# Tokenize and print all words in german_text
	all_words = word_tokenize(german_text)
	print(all_words)

	# Tokenize and print only capital words
	capital_words = r"[A-ZÜ]\w+"
	print(regexp_tokenize(german_text, capital_words))

	# Tokenize and print only emoji
	emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
	print(regexp_tokenize(german_text, emoji))


# Plotting line length with a histogram

	# Split the script into lines: lines
	lines = holy_grail.split('\n')

	# Replace all script lines for speaker
	pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
	lines = [re.sub(pattern, '', l) for l in lines]

	# Tokenize each line: tokenized_lines
	tokenized_lines = [regexp_tokenize(s,"\w+") for s in lines]

	# Make a frequency list of lengths: line_num_words
	line_num_words = [len(t_line) for t_line in tokenized_lines]

	# Plot a histogram of the line lengths
	plt.hist(line_num_words)

	# Show the plot
	plt.show()


# Creating nice bags of words
	# Import WordNetLemmatizer
	from nltk.stem import WordNetLemmatizer

	# Retain alphabetic words: alpha_only
	alpha_only = [t for t in lower_tokens if t.isalpha()]

	# Remove all stop words: no_stops
	no_stops = [t for t in alpha_only if t not in english_stops]

	# Instantiate the WordNetLemmatizer
	wordnet_lemmatizer = WordNetLemmatizer()

	# Lemmatize all tokens into a new list: lemmatized
	lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

	# Create the bag-of-words: bow
	bow = Counter(lemmatized)

	# Print the 10 most common tokens
	print(bow.most_common(10))

# Using gensim

	# Import Dictionary
	from gensim.corpora.dictionary import Dictionary

	# Create a Dictionary from the articles: dictionary
	dictionary = Dictionary(articles)

	# Select the id for "computer": computer_id
	computer_id = dictionary.token2id.get("computer")

	# Use computer_id with the dictionary to print the word
	print(dictionary.get(computer_id))

	# Create a MmCorpus: corpus
	corpus = [dictionary.doc2bow(article) for article in articles]

	# Print the first 10 word ids with their frequency counts from the fifth document
	print(corpus[4][:10])


	# Save the fifth document: doc
	doc = corpus[4]

	# Sort the doc for frequency: bow_doc
	bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

	# Print the top 5 words of the document alongside the count
	for word_id, word_count in bow_doc[:5]:
	    print(dictionary.get(word_id), word_count)
	    
	# Create the defaultdict: total_word_count
	total_word_count = defaultdict(int)
	for word_id, word_count in itertools.chain.from_iterable(corpus):
	    total_word_count[word_id] += word_count

# + Compliqué


	# Save the fifth document: doc
	doc = corpus[4]

	# Sort the doc for frequency: bow_doc
	bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

	# Print the top 5 words of the document alongside the count
	for word_id, word_count in bow_doc[:5]:
	    print(dictionary.get(word_id), word_count)
	    
	# Create the defaultdict: total_word_count
	total_word_count = defaultdict(int)
	for word_id, word_count in itertools.chain.from_iterable(corpus):
	    total_word_count[word_id] += word_count
	    
	# Create a sorted list from the defaultdict: sorted_word_count
	sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 

	# Print the top 5 words across all documents alongside the count
	for word_id, word_count in sorted_word_count[:5]:
	    print(dictionary.get(word_id), word_count)



# Using tfidf
	
	# Create a new TfidfModel using the corpus: tfidf
	tfidf = TfidfModel(corpus)

	# Calculate the tfidf weights of doc: tfidf_weights
	tfidf_weights = tfidf[doc]

	# Print the first five weights
	print(tfidf_weights[:5])

	# Sort the weights from highest to lowest: sorted_tfidf_weights
	sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

	# Print the top 5 weighted words
	for term_id, weight in sorted_tfidf_weights[:5]:
	    print(dictionary.get(term_id), weight)


# Using nltk and parts of speech


	# Tokenize the article into sentences: sentences
	sentences = sent_tokenize(article)

	# Tokenize each sentence into words: token_sentences
	token_sentences = [word_tokenize(sent) for sent in sentences]

	# Tag each tokenized sentence into parts of speech: pos_sentences
	pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

	# Create the named entity chunks: chunked_sentences
	chunked_sentences = nltk.ne_chunk_sents(pos_sentences,binary=True)

	# Test for stems of the tree with 'NE' tags
	for sent in chunked_sentences:
	    for chunk in sent:
	        if hasattr(chunk, "label") and chunk.label() == "NE":
	            print(chunk)

	# Create the defaultdict: ner_categories
	ner_categories = defaultdict(int)

	# Create the nested for loop
	for sent in chunked_sentences: # Trees
	    for chunk in sent: # Words
	        if hasattr(chunk, 'label'): # Si on a trouvé une catégorie pour le mot
	            ner_categories[chunk.label()] += 1 # On compte le nombre d'entités dans chaque catégorie
	            
	# Create a list from the dictionary keys for the chart labels: labels
	labels = list(ner_categories.keys())

	# Create a list of the values: values
	values = [ner_categories.get(v) for v in labels]

	# Create the pie chart
	plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

	# Display the chart
	plt.show()

# Introduction to spacy

# Even easier than the previous way to find categories
	# Import spacy
	import spacy

	# Instantiate the English model: nlp
	nlp = spacy.load('en',tagger=False,parser=False,matcher=False)

	# Create a new document: doc
	doc = nlp(article)

	# Print all of the found entities and their labels
	for ent in doc.ents:
	    print(ent.label_, ent.text)

# If language != "en", on utilisera polyglot

	# Create a new text object using Polyglot's Text class: txt
	txt = Text(article) # Article en français

	# Print each of the entities found
	for ent in txt.entities:
	    print(ent)
	    
	# Print the type of ent
	print(type(ent))

	# Create the list of tuples: entities
	entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

	# Print entities
	print(entities)


	# Initialize the count variable: count
	count = 0

	# Iterate over all the entities
	for ent in txt.entities:
	    # Check whether the entity contains 'Márquez' or 'Gabo'
	    if 'Márquez' in ent or 'Gabo' in ent:
	        # Increment count
	        count+=1

	# Print count
	print(count)

	# Calculate the percentage of entities that refer to "Gabo": percentage
	percentage = count / len(txt.entities)
	print(percentage)


