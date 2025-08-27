import json
import math
import re
from collections import defaultdict
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

class InvertedIndexer:
    def __init__(self):
        self.index = defaultdict(list)
        self.document_frequencies = defaultdict(int)
        self.document_vectors = {}
        self.documents = []
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stopwords and stemming"""
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        # Remove punctuation and numbers
        tokens = [token for token in tokens if token.isalnum()]
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return tokens

    def calculate_term_frequency(self, term, document_tokens):
        """Calculate TF (term frequency)"""
        return document_tokens.count(term) / len(document_tokens)

    def calculate_inverse_document_frequency(self, term):
        """Calculate IDF (inverse document frequency)"""
        return math.log(len(self.documents) / (self.document_frequencies[term] + 1))

    def build_index(self, json_file):
        """Build inverted index from crawled publications"""
        print("Loading documents...")
        with open(json_file, 'r', encoding='utf-8') as f:
            publications = json.load(f)

        print("Building inverted index...")
        for doc_id, pub in enumerate(publications):
            # Combine title and abstract for indexing
            content = f"{pub['title']} {pub['abstract']}"
            tokens = self.preprocess_text(content)
            
            # Store original document
            self.documents.append({
                'id': doc_id,
                'title': pub['title'],
                'url': pub['url'],
                'date': pub['date'],
                'authors': pub['authors'],
                'abstract': pub['abstract']
            })

            # Calculate term frequencies for document
            term_freq = defaultdict(float)
            for token in tokens:
                term_freq[token] += 1

            # Update document frequencies
            for term in set(tokens):
                self.document_frequencies[term] += 1

            # Store normalized term frequencies
            doc_length = len(tokens)
            vector = {}
            for term, freq in term_freq.items():
                tf = freq / doc_length
                self.index[term].append((doc_id, tf))
                vector[term] = tf
            
            self.document_vectors[doc_id] = vector

        # Calculate final TF-IDF weights
        print("Calculating TF-IDF weights...")
        for term in self.index:
            idf = self.calculate_inverse_document_frequency(term)
            for idx, (doc_id, tf) in enumerate(self.index[term]):
                tfidf = tf * idf
                self.index[term][idx] = (doc_id, tfidf)

        print(f"Indexed {len(self.documents)} documents with {len(self.index)} unique terms.")

    def save_index(self, filename='inverted_index.pkl'):
        """Save the inverted index to a file"""
        data = {
            'index': dict(self.index),
            'document_frequencies': dict(self.document_frequencies),
            'document_vectors': self.document_vectors,
            'documents': self.documents
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index saved to {filename}")

    def load_index(self, filename='inverted_index.pkl'):
        """Load the inverted index from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.index = defaultdict(list, data['index'])
        self.document_frequencies = defaultdict(int, data['document_frequencies'])
        self.document_vectors = data['document_vectors']
        self.documents = data['documents']
        print(f"Index loaded from {filename}")

def main():
    # Create and build the inverted index
    indexer = InvertedIndexer()
    indexer.build_index('coventry_publications.json')
    indexer.save_index()

if __name__ == '__main__':
    main()