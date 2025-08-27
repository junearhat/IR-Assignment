import math
from collections import defaultdict
from indexer import InvertedIndexer
import numpy as np

class DocumentRanker:
    def __init__(self, indexer: InvertedIndexer):
        self.indexer = indexer

    def calculate_query_vector(self, query_terms):
        """Calculate TF-IDF vector for query terms"""
        query_vector = defaultdict(float)
        query_length = len(query_terms)
        
        # Calculate term frequencies in query
        for term in query_terms:
            query_vector[term] += 1.0 / query_length
            
        # Apply IDF weights
        for term in query_vector:
            idf = self.indexer.calculate_inverse_document_frequency(term)
            query_vector[term] *= idf
            
        return query_vector

    def calculate_cosine_similarity(self, query_vector, document_vector):
        """Calculate cosine similarity between query and document vectors"""
        # Find common terms
        common_terms = set(query_vector.keys()) & set(document_vector.keys())
        
        if not common_terms:
            return 0.0
            
        # Calculate numerator (dot product)
        numerator = sum(query_vector[term] * document_vector[term] 
                       for term in common_terms)
        
        # Calculate denominators (vector magnitudes)
        query_magnitude = math.sqrt(sum(weight ** 2 
                                      for weight in query_vector.values()))
        doc_magnitude = math.sqrt(sum(weight ** 2 
                                    for weight in document_vector.values()))
        
        # Avoid division by zero
        if query_magnitude == 0.0 or doc_magnitude == 0.0:
            return 0.0
            
        return numerator / (query_magnitude * doc_magnitude)

    def rank_documents(self, query):
        """Rank documents based on query relevance using cosine similarity"""
        # Preprocess query
        query_terms = self.indexer.preprocess_text(query)
        if not query_terms:
            return []
            
        # Calculate query vector
        query_vector = self.calculate_query_vector(query_terms)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_vector in self.indexer.document_vectors.items():
            similarity = self.calculate_cosine_similarity(query_vector, doc_vector)
            similarities.append((doc_id, similarity))
        
        # Sort by similarity score in descending order
        ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Return top-k results with document information
        results = []
        for doc_id, score in ranked_docs:
            doc_info = self.indexer.documents[doc_id].copy()
            doc_info['score'] = round(score, 4)
            results.append(doc_info)
            
        return results

def main():
    # Load the index
    indexer = InvertedIndexer()
    indexer.load_index()
    
    # Create ranker
    ranker = DocumentRanker(indexer)
    
    # Example query
    query = input("Enter your search query: ")
    results = ranker.rank_documents(query)
    
    # Print results
    print(f"\nTop 5 results for query: '{query}'\n")
    for rank, doc in enumerate(results, 1):
        print(f"{rank}. {doc['title']}")
        print(f"   Score: {doc['score']}")
        print(f"   Authors: {', '.join(author['name'] for author in doc['authors'])}")
        if 'abstract' in doc:
            print(f"   Abstract: {doc['abstract']}")
        print(f"   URL: {doc['url']}")
        print(f"   Date: {doc['date']}")
        print()

if __name__ == '__main__':
    main()