import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
from nltk.util import ngrams
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Download necessary NLTK data
nltk.download('punkt')

class SentenceEmbeddingModel:
    def _init_(self, ngram_range=(1, 5), vector_size=100):
        self.ngram_range = ngram_range
        self.vector_size = vector_size
        self.model = None

    def preprocess(self, text):
        """Tokenizes the input text."""
        tokenizer = TreebankWordTokenizer()
        return tokenizer.tokenize(text.lower())

    def generate_ngrams(self, sentence):
        """Generates n-grams for the sentence based on the ngram_range."""
        tokens = self.preprocess(sentence)
        ngrams_list = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams_list += list(ngrams(tokens, n))
        return [' '.join(gram) for gram in ngrams_list]

    def train_word2vec(self, sentences, alpha=0.025):
        """Train a Word2Vec model on the dataset with adjustable learning rate."""
        tokenized_sentences = [self.preprocess(sentence) for sentence in sentences]
        self.model = Word2Vec(sentences=tokenized_sentences, vector_size=self.vector_size, window=5, min_count=1, sg=1, alpha=alpha)
        self.model.train(tokenized_sentences, total_examples=len(sentences), epochs=10)


    def get_ngram_embedding(self, ngram):
        """Get the embedding for an n-gram by averaging the embeddings of its words."""
        tokens = ngram.split()
        token_vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
        if token_vectors:
            return np.mean(token_vectors, axis=0)
        return None

    def get_sentence_embedding(self, sentence):
        """Compose a sentence embedding by averaging all n-gram embeddings."""
        ngrams_list = self.generate_ngrams(sentence)
        ngram_embeddings = [self.get_ngram_embedding(ngram) for ngram in ngrams_list]
        ngram_embeddings = [emb for emb in ngram_embeddings if emb is not None]
        if ngram_embeddings:
            return np.mean(ngram_embeddings, axis=0)
        return np.zeros(self.vector_size)

    def compute_embeddings(self, sentences):
        """Compute embeddings for a list of sentences."""
        return {sentence: self.get_sentence_embedding(sentence) for sentence in tqdm(sentences)}

    def contrastive_loss(self, sentence_embedding, positive_embedding, negative_embedding, margin=0.1):
        """Compute margin-based contrastive loss."""
        positive_sim = cosine_similarity([sentence_embedding], [positive_embedding])[0][0]
        negative_sim = cosine_similarity([sentence_embedding], [negative_embedding])[0][0]
        return max(0, negative_sim - positive_sim + margin)

    def perform_contrastive_learning(self, sentences, epochs=20, margin=0.1):
        """Perform unsupervised contrastive learning with adjustable number of epochs."""
        sentence_embeddings = self.compute_embeddings(sentences)

        for epoch in range(epochs):
            total_loss = 0
            for sentence in sentences:
                positive_idx = random.choice([i for i in range(len(sentences)) if sentences[i] != sentence])
                negative_idx = random.choice([i for i in range(len(sentences)) if i != positive_idx])

                sentence_embedding = sentence_embeddings[sentence]
                positive_embedding = sentence_embeddings[sentences[positive_idx]]
                negative_embedding = sentence_embeddings[sentences[negative_idx]]

                loss = self.contrastive_loss(sentence_embedding, positive_embedding, negative_embedding, margin)
                total_loss += loss

            avg_loss = total_loss / len(sentences)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def evaluate_text_classification(self, sentences, labels):
            """Evaluate embeddings using a text classification task."""
            # Compute embeddings and store them in a list
            sentence_embeddings = [self.get_sentence_embedding(sentence) for sentence in sentences]
            X = np.array(sentence_embeddings)
            y = np.array(labels)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            print(classification_report(y_test, y_pred))



    # Intrinsic evaluation function for cosine similarity between sentence pairs
    def evaluate_embedding_quality(self, sentence_pairs):
        """Evaluates sentence similarity using cosine similarity between embeddings."""
        for sent1, sent2 in sentence_pairs:
            embedding1 = self.get_sentence_embedding(sent1)
            embedding2 = self.get_sentence_embedding(sent2)
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            print(f"Similarity between \"{sent1}\" and \"{sent2}\": {similarity:.4f}")


# Function to load dataset from an uploaded file in Google Colab
def load_dataset_from_upload():
    uploaded = files.upload()  # This will prompt you to upload a file in Colab
    for filename in uploaded.keys():
        # Open and read the file with ISO-8859-1 encoding
        with open(filename, 'r', encoding='ISO-8859-1') as file:
            data = file.readlines()
    return [line.strip() for line in data if line.strip()]


if _name_ == "_main_":
    # Step 1: Upload dataset file in Google Colab
    print("Please upload your dataset file (txt format)")
    sentences = load_dataset_from_upload()

    # Step 2: Define labels for the sentences (for demonstration purposes, use dummy labels)
    # Ensure you have labels for your sentences
    labels = [random.choice([0, 1]) for _ in range(len(sentences))]  # Replace with actual labels

    # Initialize the sentence embedding model with custom parameters
    sentence_embedding_model = SentenceEmbeddingModel(ngram_range=(1, 3), vector_size=100)

    # Step 3: Train Word2Vec model on the uploaded dataset
    sentence_embedding_model.train_word2vec(sentences)

    # Step 4: Perform unsupervised contrastive learning
    sentence_embedding_model.perform_contrastive_learning(sentences, epochs=5)

    # Step 5: Evaluate the model using a downstream task (text classification in this case)
    sentence_embedding_model.evaluate_text_classification(sentences, labels)
