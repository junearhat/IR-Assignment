import tkinter as tk
from tkinter import scrolledtext
import webbrowser
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# Assuming InvertedIndexer and DocumentRanker classes are defined elsewhere
from indexer import InvertedIndexer
from ranker import DocumentRanker

# ----------------------------------------------------------------------------------------------------------------------

class SearchEngine:
    def __init__(self):
        # Initialize the indexer and ranker
        self.indexer = InvertedIndexer()
        self.indexer.load_index()
        self.ranker = DocumentRanker(self.indexer)

        # Create main window using ttkbootstrap's style
        self.window = ttk.Window(title="Research Publication Search Engine", themename="cosmo") 
        self.window.geometry("800x600")

        # Create and pack a themed frame for the search bar
        search_frame = ttk.Frame(self.window, padding=15)
        search_frame.pack(fill=tk.X, padx=10, pady=10)

        # Search entry with a modern look
        self.search_entry = ttk.Entry(search_frame, bootstyle="info", width=60, font=('Helvetica', 12))
        self.search_entry.pack(side=tk.LEFT, padx=(0, 5), ipady=4) 

        # Search button with a modern, colored style
        search_button = ttk.Button(search_frame, text="Search", bootstyle="info, outline", command=self.perform_search)
        search_button.pack(side=tk.LEFT, padx=5)

        # Create a frame for the results area to add a border
        results_frame = ttk.Frame(self.window, padding=10, relief=tk.SUNKEN, borderwidth=1)
        results_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Results area with improved font and a modern look
        self.results_area = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, font=('Arial', 10), bd=0, relief=tk.FLAT)
        self.results_area.pack(fill=tk.BOTH, expand=True)

        # Bind the Enter key to the search function
        self.search_entry.bind('<Return>', lambda e: self.perform_search())

    def perform_search(self):
        query = self.search_entry.get().strip()
        if not query:
            return

        self.results_area.configure(state='normal')
        self.results_area.delete(1.0, tk.END)

        # Perform the search using the DocumentRanker instance
        results = self.ranker.rank_documents(query)

        # Filter results with score > 0.1
        filtered_results = [doc for doc in results if doc['score'] > 0.1]

        if not filtered_results:
            self.results_area.insert(tk.END, "No results found with sufficient relevance.")
            self.results_area.configure(state='disabled')
            return

        # Define and apply styles for better visual hierarchy
        self.results_area.tag_configure("title", font=("Helvetica", 14, "bold"), foreground="#2C3E50") 
        self.results_area.tag_configure("authors", font=("Arial", 10, "italic"), foreground="#7F8C8D")
        self.results_area.tag_configure("info", font=("Arial", 9), foreground="#34495E")
        self.results_area.tag_configure("abstract", font=("Arial", 10), foreground="#555555")
        self.results_area.tag_configure("url", font=("Arial", 10, "underline"), foreground="#3498DB")

        for i, doc in enumerate(filtered_results, 1):
            # Title with ranking
            self.results_area.insert(tk.END, f"{i}. {doc['title']}\n", "title")

            # Authors (make clickable if author has a URL)
            authors_start = self.results_area.index(tk.END)
            author_tags = []
            for j, author in enumerate(doc['authors']):
                name = author['name']
                if 'url' in author and author['url']:
                    tag = f"author_{i}_{j}"
                    author_tags.append((tag, author['url']))
                    self.results_area.insert(tk.END, name, tag)
                    # Style clickable author
                    self.results_area.tag_configure(tag, foreground="#1ABC9C", underline=True, font=("Arial", 10, "italic"))
                    # Bind click event
                    self.results_area.tag_bind(tag, "<Button-1>", lambda e, url=author['url']: webbrowser.open(url))
                    self.results_area.tag_bind(tag, "<Enter>", lambda e: self.results_area.config(cursor="hand2"))
                    self.results_area.tag_bind(tag, "<Leave>", lambda e: self.results_area.config(cursor=""))
                else:
                    self.results_area.insert(tk.END, name, "authors")
                if j < len(doc['authors']) - 1:
                    self.results_area.insert(tk.END, ", ", "authors")
            self.results_area.insert(tk.END, "\n", "authors")

            # Score and date
            self.results_area.insert(tk.END, f"Relevance Score: {doc['score']:.2f} | Date: {doc['date']}\n", "info")

            # Abstract
            abstract = doc['abstract']
            if len(abstract) > 200:
                abstract = abstract[:200] + "..."
            self.results_area.insert(tk.END, f"Abstract: {abstract}\n\n", "abstract")

            # URL
            url_start = self.results_area.index("end-1c linestart")
            self.results_area.insert(tk.END, f"URL: {doc['url']}\n", "url")
            url_end = self.results_area.index("end-1c")

            # Create unique tag for this URL
            url_tag = f"url_{i}"
            self.results_area.tag_add(url_tag, url_start, url_end)
            self.results_area.tag_add("url", url_start, url_end)  # Keep original styling
            
            # Bind click event with unique tag
            self.results_area.tag_bind(url_tag, "<Button-1>", 
                                     lambda e, url=doc['url']: webbrowser.open(url))
            
            # Update hover effects for the unique tag
            self.results_area.tag_bind(url_tag, "<Enter>", 
                                     lambda e: self.results_area.config(cursor="hand2"))
            self.results_area.tag_bind(url_tag, "<Leave>", 
                                     lambda e: self.results_area.config(cursor=""))

            # Separator with a subtle, modern color
            self.results_area.insert(tk.END, "\n" + "_"*80 + "\n\n", "separator")

        self.results_area.configure(state='disabled') # Make results read-only

    def run(self):
        self.window.mainloop()

def main():
    search_engine = SearchEngine()
    search_engine.run()

if __name__ == "__main__":
    main()