# Main Script for Document Classification System with GUI

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Download required NLTK data if not already present
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK data (stopwords, punkt, wordnet)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("Downloads complete.")

class DocumentClassifier:
    """A comprehensive document classification system."""
    def __init__(self):
        # The pipeline is constructed with an *unfitted* LinearSVC inside the CalibratedClassifierCV
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(sublinear_tf=True)), 
            ('select', SelectKBest(chi2)),
            ('clf', CalibratedClassifierCV(
                LinearSVC(class_weight='balanced', max_iter=10000, dual=False), 
                cv=3
            ))
        ])
        self.best_estimator_ = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        """Cleans and prepares text for classification."""
        if not isinstance(text, str):
            return ""
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in self.stop_words and word.isalpha()]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    def load_data(self, csv_file):
        """Loads and preprocesses the dataset from a CSV file."""
        df = pd.read_csv(csv_file)
        df['text_to_process'] = df['title'].fillna('') + " " + df['summary'].fillna('')
        df['processed_text'] = df['text_to_process'].apply(self.preprocess)
        df.dropna(subset=['category'], inplace=True)
        return df

    def optimize_hyperparameters(self, X, y, cv_splits=5):
        """Performs a grid search to find the best hyperparameters for the pipeline."""
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.85, 0.95],
            'select__k': [3000, 5000, 'all'],
            'clf__estimator__C': [0.1, 1, 10] 
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        gs = GridSearchCV(self.pipeline, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=1)
        print("Starting hyperparameter search...")
        gs.fit(X, y)
        print(f"\nBest grid-search params: {gs.best_params_}")
        print(f"Best cross-validated F1-macro score: {gs.best_score_:.4f}")
        self.best_estimator_ = gs.best_estimator_

    def train(self, X, y, tune=False):
        """Trains the classifier. If tune=True, performs GridSearchCV first."""
        if tune:
            self.optimize_hyperparameters(X, y)
        else:
            print("Training model with default parameters...")
            self.pipeline.fit(X, y)
            self.best_estimator_ = self.pipeline
        print("Training complete.")
        
    def evaluate_full(self, X, y):
        """Evaluates the classifier on the entire dataset and shows confusion matrix."""
        if self.best_estimator_ is None:
            print("Model is not trained yet. Please call train() first.")
            return

        print("\n--- Full Dataset Evaluation ---")
        predictions = self.best_estimator_.predict(X)

        print("\nClassification Report (Full Data):")
        print(classification_report(y, predictions, digits=4))

        acc = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='macro')
        print(f"Overall Accuracy: {acc:.4f}")
        print(f"Overall Macro F1-Score: {f1:.4f}\n")

        labels = sorted(list(set(y)))
        cm = confusion_matrix(y, predictions, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix (Full Dataset)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def predict(self, text):
        """Predicts the category for a new piece of text and returns its confidence."""
        if self.best_estimator_ is None:
            raise RuntimeError("Model has not been trained. Please train the model before prediction.")
            
        processed_text = self.preprocess(text)
        probs = self.best_estimator_.predict_proba([processed_text])[0]
        
        pred_idx = np.argmax(probs)
        pred_label = self.best_estimator_.classes_[pred_idx]
        confidence = probs[pred_idx]
        
        return pred_label, confidence

    def show_top_features(self, n=20):
        """Displays the top N most important features for each class."""
        if self.best_estimator_ is None or not hasattr(self.best_estimator_, 'named_steps'):
            print("Cannot show features. The model is not a trained pipeline.")
            return
            
    
        
        tfidf = self.best_estimator_.named_steps['tfidf']
        select = self.best_estimator_.named_steps['select']
        calibrated_clf = self.best_estimator_.named_steps['clf']

        if not hasattr(calibrated_clf, 'calibrated_classifiers_') or not calibrated_clf.calibrated_classifiers_:
            print("The classifier has not been fitted yet.")
            return
        
        clf = calibrated_clf.calibrated_classifiers_[0]

        if not hasattr(clf, 'coef_'):
            return

        feature_names = np.array(tfidf.get_feature_names_out())
        mask = select.get_support()
        selected_feature_names = feature_names[mask]
        
        for i, class_label in enumerate(self.best_estimator_.classes_):
            coef_index = i if clf.coef_.shape[0] > 1 else 0
            top_n_indices = np.argsort(clf.coef_[coef_index])[-n:][::-1]
            top_features = selected_feature_names[top_n_indices]
            print(f"'{class_label}': {', '.join(top_features)}")
        print("-" * 30)


class ClassifierGUI:
    """GUI for the Document Classification System."""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.root = tk.Tk()
        self.root.title("Document Classification System")
        self.root.geometry("900x800")
        self.root.configure(bg='#f0f0f0')
        
        # Store training data for evaluation
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Store evaluation results to avoid recomputing
        self.evaluation_results = None
        
        # Configure matplotlib to use Agg backend for thread safety
        import matplotlib
        matplotlib.use('Agg')
        
        # Configure style for better appearance
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles with larger fonts
        self.style.configure('Title.TLabel', 
                           font=('Arial', 24, 'bold'),
                           foreground='#2c3e50',
                           background='#f0f0f0')
        
        self.style.configure('Heading.TLabel',
                           font=('Arial', 16, 'bold'),
                           foreground='#34495e',
                           background='#f0f0f0')
        
        self.style.configure('Result.TLabel',
                           font=('Arial', 14),
                           foreground='#2c3e50',
                           background='#ecf0f1',
                           relief='solid',
                           borderwidth=1)
        
        self.style.configure('Custom.TButton',
                           font=('Arial', 14, 'bold'),
                           foreground='white',
                           background='#3498db')
        
        # Configure label frame fonts
        self.style.configure('TLabelframe.Label',
                           font=('Arial', 14, 'bold'))
        
        self.style.configure('TLabel',
                           font=('Arial', 13))
        
        self.style.map('Custom.TButton',
                      background=[('active', '#2980b9'),
                                ('pressed', '#21618c')])
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="25")  # Increased padding
        main_frame.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="📄 Document Classification System",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 25))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="15")  # Increased padding
        status_frame.grid(row=1, column=0, sticky=tk.W+tk.E, pady=(0, 25))
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Model Status:", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Not Trained", foreground='red', font=('Arial', 14))  # Added explicit font
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(15, 0))
        
        # Train button
        self.train_button = ttk.Button(status_frame, 
                                     text="🚀 Train Model",
                                     command=self.train_model,
                                     style='Custom.TButton')
        self.train_button.grid(row=0, column=2, padx=(15, 0))
        
        # Evaluation button (initially hidden)
        self.eval_button = ttk.Button(status_frame, 
                                    text="📊 Show Evaluation",
                                    command=self.show_evaluation,
                                    style='Custom.TButton',
                                    state='disabled')
        self.eval_button.grid(row=0, column=3, padx=(15, 0))
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Text Classification", padding="20")  # Increased padding
        input_frame.grid(row=2, column=0, sticky=tk.W+tk.E+tk.N+tk.S, pady=(0, 25))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)
        
        ttk.Label(input_frame, 
                 text="Enter your text below for classification:",
                 style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        
        # Text input area with larger font
        self.text_input = scrolledtext.ScrolledText(input_frame, 
                                                   height=8, 
                                                   width=60,  # Adjusted width for larger font
                                                   font=('Arial', 16),  # Increased from 11
                                                   bg='white',
                                                   fg='black',
                                                   insertbackground='black',
                                                   wrap=tk.WORD)
        self.text_input.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N+tk.S, pady=(0, 20))
        
        # Classify button
        self.classify_button = ttk.Button(input_frame, 
                                        text="🔍 Classify Text",
                                        command=self.classify_text,
                                        style='Custom.TButton',
                                        state='disabled')
        self.classify_button.grid(row=2, column=0, pady=(0, 20))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="20")  # Increased padding
        results_frame.grid(row=3, column=0, sticky=tk.W+tk.E, pady=(0, 25))
        results_frame.columnconfigure(1, weight=1)
        
        # Predicted category
        ttk.Label(results_frame, text="Predicted Category:", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        self.category_result = ttk.Label(results_frame, 
                                       text="---",
                                       style='Result.TLabel',
                                       padding="15")  # Increased padding
        self.category_result.grid(row=0, column=1, sticky=tk.W+tk.E, padx=(15, 0), pady=(0, 15))
        
        # Confidence
        ttk.Label(results_frame, text="Confidence:", style='Heading.TLabel').grid(row=1, column=0, sticky=tk.W, pady=(0, 15))
        self.confidence_result = ttk.Label(results_frame, 
                                         text="---",
                                         style='Result.TLabel',
                                         padding="15")  # Increased padding
        self.confidence_result.grid(row=1, column=1, sticky=tk.W+tk.E, padx=(15, 0), pady=(0, 15))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, sticky=tk.W+tk.E, pady=(0, 15))
        
        # Footer with larger font
        footer_label = ttk.Label(main_frame, 
                               text="© 2025 Document Classification System",
                               font=('Arial', 12),  # Increased from 9
                               foreground='#7f8c8d')
        footer_label.grid(row=5, column=0, pady=(15, 0))
        
        # Configure main frame row weights
        main_frame.rowconfigure(2, weight=1)

    def train_model(self):
        """Train the model in a separate thread."""
        def train_worker():
            try:
                self.root.after(0, self.show_training_progress)
                
                print("Loading and preprocessing dataset...")
                df = self.classifier.load_data('news_article.csv')
                print(f"Dataset loaded successfully with {len(df)} records.")
                
                X = df['processed_text']
                y = df['category']
                
                # Store for later evaluation
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train the model
                self.classifier.train(self.X_train, self.y_train, tune=True)
                
                # Compute evaluation results (but don't show plots yet)
                self.compute_evaluation_results()
                
                self.root.after(0, self.training_complete)
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.training_error(error_msg))
        
        # Start training in background thread
        training_thread = threading.Thread(target=train_worker)
        training_thread.daemon = True
        training_thread.start()
    
    def compute_evaluation_results(self):
        """Compute evaluation results without displaying plots."""
        if self.classifier.best_estimator_ is None or self.X_test is None or self.y_test is None:
            return
        
        # Evaluate on test set
        y_pred = self.classifier.best_estimator_.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        
        # Get classification report
        class_report = classification_report(self.y_test, y_pred, digits=4)
        
        # Get confusion matrix data
        labels = sorted(list(set(self.y_test)))
        cm = confusion_matrix(self.y_test, y_pred, labels=labels)
        
        # Full dataset evaluation
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = pd.concat([self.y_train, self.y_test])
        y_full_pred = self.classifier.best_estimator_.predict(X_full)
        
        full_accuracy = accuracy_score(y_full, y_full_pred)
        full_f1 = f1_score(y_full, y_full_pred, average='macro')
        
        # Store results
        self.evaluation_results = {
            'test_accuracy': accuracy,
            'test_f1': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'labels': labels,
            'full_accuracy': full_accuracy,
            'full_f1': full_f1,
            'y_test': self.y_test,
            'y_pred': y_pred
        }
        
        # Print console results
        print("\n" + "="*60)
        print("          MODEL EVALUATION RESULTS")
        print("="*60)
        print("\n📊 TEST SET METRICS:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   F1 Score (Macro): {f1:.4f}")
        print("\n📋 CLASSIFICATION REPORT:")
        print("-" * 50)
        print(class_report)
        print("\n📈 FULL DATASET EVALUATION:")
        print(f"   Full Dataset Accuracy: {full_accuracy:.4f} ({full_accuracy*100:.2f}%)")
        print(f"   Full Dataset F1 Score: {full_f1:.4f}")
        print("-" * 50)
        self.classifier.show_top_features(n=10)
        print("="*60)
    
    def show_evaluation(self):
        """Show evaluation results with plots on main thread."""
        if self.evaluation_results is None:
            messagebox.showwarning("No Evaluation Data", "Please train the model first.")
            return
        
        try:
            # Switch back to interactive backend for main thread
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get stored results
            results = self.evaluation_results
            
            # Create confusion matrix for FULL dataset instead of test set
            X_full = pd.concat([self.X_train, self.X_test])
            y_full = pd.concat([self.y_train, self.y_test])
            y_full_pred = self.classifier.best_estimator_.predict(X_full)
            
            # Generate confusion matrix for full dataset
            full_cm = confusion_matrix(y_full, y_full_pred, labels=results['labels'])
            
            # Create and show confusion matrix on main thread
            plt.figure(figsize=(12, 10))
            sns.heatmap(full_cm, 
                       annot=True, 
                       fmt='d', 
                       cmap='Blues', 
                       xticklabels=results['labels'], 
                       yticklabels=results['labels'],
                       cbar_kws={'label': 'Number of Samples'})
            
            plt.title('Confusion Matrix - Full Dataset Evaluation', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('True Label', fontsize=12, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
            
            # Show evaluation summary in a message box
            summary = (f"📊 MODEL EVALUATION SUMMARY\n\n"
                      f"Test Set Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)\n"
                      f"Test Set F1 Score: {results['test_f1']:.4f}\n\n"
                      f"Full Dataset Accuracy: {results['full_accuracy']:.4f} ({results['full_accuracy']*100:.2f}%)\n"
                      f"Full Dataset F1 Score: {results['full_f1']:.4f}\n\n"
                      f"Confusion Matrix shows: Full Dataset Results")
            
            messagebox.showinfo("Evaluation Results", summary)
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to show evaluation plots:\n{str(e)}")
    
    def show_training_progress(self):
        """Show training progress."""
        self.progress.start(10)
        self.train_button.configure(state='disabled', text='Training...')
        self.status_label.configure(text="Training in progress...", foreground='orange')
    
    def training_complete(self):
        """Handle training completion."""
        self.progress.stop()
        self.train_button.configure(state='normal', text='✅ Retrain Model')
        self.classify_button.configure(state='normal')
        self.eval_button.configure(state='normal')
        self.status_label.configure(text="Model Ready", foreground='green')
        messagebox.showinfo("Training Complete", 
                          "Model has been trained successfully!\n\n"
                          "✅ Accuracy and F1 scores calculated\n"
                          "✅ Classification report generated\n"
                          "✅ Confusion matrix computed\n\n"
                          "Click 'Show Evaluation' to see the confusion matrix plot.\n"
                          "Check the console for detailed results!")
    
    def training_error(self, error_msg):
        """Handle training error."""
        self.progress.stop()
        self.train_button.configure(state='normal', text='🚀 Train Model')
        self.status_label.configure(text="Training Failed", foreground='red')
        messagebox.showerror("Training Error", f"Failed to train model:\n{error_msg}")
    
    def classify_text(self):
        """Classify the input text."""
        input_text = self.text_input.get("1.0", tk.END).strip()
        
        if not input_text:
            messagebox.showwarning("Empty Input", "Please enter some text to classify.")
            return
        
        try:
            self.progress.start(10)
            self.classify_button.configure(state='disabled', text='Classifying...')
            
            # Perform classification
            prediction, confidence = self.classifier.predict(input_text)
            
            # Update results
            self.category_result.configure(text=f"📂 {prediction.title()}")
            confidence_text = f"🎯 {confidence*100:.2f}%"
            
            self.confidence_result.configure(text=confidence_text)
            
            self.progress.stop()
            self.classify_button.configure(state='normal', text='🔍 Classify Text')
            
        except Exception as e:
            self.progress.stop()
            self.classify_button.configure(state='normal', text='🔍 Classify Text')
            messagebox.showerror("Classification Error", f"Failed to classify text:\n{str(e)}")
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()


def main():
    """Main function to run the document classification workflow."""
    print("Initializing Document Classification System...")
    classifier = DocumentClassifier()
    
    # Check if we want to run GUI or console mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--console':
        # Console mode (original functionality)
        print("Running in console mode...")
        
        print("Loading and preprocessing dataset...")
        df = classifier.load_data('news_article.csv')
        print(f"Dataset loaded successfully with {len(df)} records.")
        
        X = df['processed_text']
        y = df['category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        classifier.train(X_train, y_train, tune=True)
        
        classifier.evaluate_full(X, y)
        
        classifier.show_top_features(n=15)
        
        print("\n--- Interactive Classification ---")
        print("Enter a news title or summary to classify. Type 'exit' to quit.")
        
        try:
            while True:
                user_text = input("> ").strip()
                if user_text.lower() == 'exit':
                    print("Exiting interactive mode.")
                    break
                if not user_text:
                    continue
                    
                prediction, confidence = classifier.predict(user_text)
                print(f"-> Predicted Category: '{prediction}' (Confidence: {confidence*100:.2f}%)\n")
        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted. Exiting.")
    else:
        # GUI mode (default)
        print("Starting GUI mode...")
        app = ClassifierGUI(classifier)
        app.run()

if __name__ == "__main__":
    main()