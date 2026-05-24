import kagglehub
import os
import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
stop_words = set(stopwords.words('english'))


# pobranie i wczytanie danych
def load_data():
    path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")
    file_path = os.path.join(path, 'Combined Data.csv')
    df = pd.read_csv(file_path)

    df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna(subset=['statement']).reset_index(drop=True)

    # podział: tekst do klasyfikacji (X), etykieta (y)
    X = df['statement']
    y = df['status']

    return df, X, y


# podział danych na train, validation i test
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42,
        shuffle=True
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
        shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# podstawowy preprocessing: lowercase + whitespace cleanup
def preprocess(text):
    text = text.lower()
    text = ' '.join(text.split())
    return text


# POS-aware lemmatizer
def tag_to_name(pos):
    pos = pos.lower()
    if pos.startswith('jj'):
        return ADJ
    elif pos.startswith('vb') or pos=='md':
        return VERB
    elif pos.startswith('rb') or pos=='wrb':
        return ADV
    else:
        return NOUN

def nltk_pos_tagger(tokens):
    tagged = nltk.pos_tag(tokens) 
    tagged = [(word, tag_to_name(tag)) for (word, tag) in tagged]
    return tagged

def pos_lemmatize(tokens):
    tags_dict = {
        NOUN : 'n',
        VERB : 'v',
        ADJ  : 'a',
        ADV  : 'r'
    }

    tagged = nltk_pos_tagger(tokens)
    lemmatizer = WordNetLemmatizer()

    lemmas = []
    for word, tag in tagged:
        word_pos = tags_dict.get(tag, 'n')
        lemma = lemmatizer.lemmatize(word, pos=word_pos)
        lemmas.append(lemma)
    return lemmas


# preprocessing v1: 
# lematyzacja + usunięcie stopwords
def preprocess_v1(text):
    text = text.lower()
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return ' '.join(tokens)


# preprocessing v2: 
# lematyzacja + usunięcie stopwords + usunięcie tokenów niealfabetycznych
def preprocess_v2(text):
    text = text.lower()
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
        and word.isalpha()
    ]

    return ' '.join(tokens)


# preprocessing v3:
# lematyzacja uwzględniająca części mowy + usunięcie stopwords
def preprocess_v3(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]
    tokens = pos_lemmatize(tokens)

    return ' '.join(tokens)


# preprocessing v4
def preprocess_v4(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    tokens = [
        word for word in tokens
        if word not in stop_words
        and word.isalpha()
    ]
    tokens = pos_lemmatize(tokens)

    return ' '.join(tokens)


# przycięcie tekstu do max. k znaków
def truncate_text(text, max_chars):
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space == -1:
        return truncated

    return truncated[:last_space]


# eksperyment z różną liczbą cech (max_features) dla TF-IDF (Logistic Regression)
def max_features(X_train, X_val, y_train, y_val):
    Fs = [500*n for n in range(1, 11)]
    Fs.append(None)

    matrixes = {}
    f1_scores = {}

    for F in Fs:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=F)),
            ("log_reg", LogisticRegression(max_iter=1_000))
        ])
        matrix = pipeline.fit(X_train, y_train)

        y_val_pred = pipeline.predict(X_val)
        f1_macro = f1_score(y_val, y_val_pred, average='macro')
        print(f"max_features={F}, F1 macro: {f1_macro:.2f}")

        matrixes[F] = matrix
        f1_scores[F] = f1_macro

    return matrixes, f1_scores


# eksperyment z różną liczbą cech (max_features) dla TF-IDF (SVM)
def max_features_svm(X_train, X_val, y_train, y_val):
    Fs = [500*n for n in range(1, 11)]
    Fs.append(None)

    f1_scores = {}

    for F in Fs:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=F)),
            ("clf", LinearSVC())
        ])
        pipeline.fit(X_train, y_train)

        y_val_pred = pipeline.predict(X_val)
        f1_macro = f1_score(y_val, y_val_pred, average='macro')
        print(f"max_features={F}, F1 macro: {f1_macro:.2f}")

        f1_scores[F] = f1_macro

    return f1_scores


# przygodowanie do wizualizacji wyników eksperyment 
# z różną liczbą cech (max_features) dla TF-IDF
def prepare_plot_data(scores_dict):
    x = []
    y = []
    
    for k, v in scores_dict.items():
        if k is None:
            x.append(5500)
        else:
            x.append(k)
        y.append(v)
    
    return x, y


# funkcja porównująca błędy różnych modeli
def compare_errors(predictions, y_true):
    results = {}
    error_sets = {}

    for name, y_pred in predictions.items():
        errors = {
            idx
            for idx, (yt, yp) in enumerate(zip(y_true, y_pred))
            if yt != yp
        }
        error_sets[name] = errors

    for model_a, model_b in combinations(predictions.keys(), 2):
        errors_a = error_sets[model_a]
        errors_b = error_sets[model_b]

        intersection = errors_a & errors_b
        union = errors_a | errors_b

        jaccard = (
            len(intersection) / len(union)
            if len(union) > 0 else 0
        )

        key = tuple(sorted([model_a, model_b]))
        results[key] = {
            "jaccard": jaccard,
            "shared_errors": sorted(intersection),
            "only_a_errors": sorted(errors_a - errors_b),
            "only_b_errors": sorted(errors_b - errors_a)
        }

    metrics = {}
    for name, y_pred in predictions.items():
        metrics[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(
                y_true,
                y_pred,
                average='macro'
            )
        }

    return results, metrics


# wizualizacja porównania błędów
def plot_error_matrix(results, metrics, metric='f1_macro'):
    model_names = list(metrics.keys())
    n = len(model_names)
    matrix = np.zeros((n, n))

    for i, model_a in enumerate(model_names):
        for j, model_b in enumerate(model_names):
            if i == j:
                matrix[i, j] = metrics[model_a][metric]
            else:
                pair = tuple(sorted([model_a, model_b]))
                if pair in results:
                    matrix[i, j] = results[pair]['jaccard']

    plt.figure(figsize=(8, 8))
    im = plt.imshow(matrix)
    plt.colorbar(im)
    plt.xticks(range(n), model_names, rotation=45)
    plt.yticks(range(n), model_names)

    for i in range(n):
        for j in range(n):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha='center',
                va='center'
            )

    plt.title(
        f"Error Similarity Matrix ({metric} on diagonal)"
    )
    plt.tight_layout()
    plt.show()


# wypisanie przykładowych błędów
def print_error_examples(
    comparison_results,
    X,
    y_true,
    predictions,
    n_examples=5
):

    for (model_a, model_b), data in comparison_results.items():
        print("\n" + "="*80)
        print(f"{model_a} vs {model_b}")
        print("="*80)

        categories = [
            ("shared_errors", "Shared errors"),
            ("only_a_errors", f"Only {model_a} errors"),
            ("only_b_errors", f"Only {model_b} errors")
        ]

        for key, title in categories:
            print(f"\n--- {title} ---")
            indices = data[key][:n_examples]

            for idx in indices:
                print(f"\nID: {idx}")
                print(f"TEXT: {X.iloc[idx][:300]}")
                print(f"TRUE: {y_true.iloc[idx]}")

                print(
                    f"{model_a}: "
                    f"{predictions[model_a][idx]}"
                )
                print(
                    f"{model_b}: "
                    f"{predictions[model_b][idx]}"
                )


# błędy/poprawne przykłady dla wszystkich modeli
def analyze_common_cases(
    predictions,
    y_true,
    X,
    k_examples=5
):

    model_names = list(predictions.keys())
    n = len(y_true)

    correct_matrix = []
    all_correct = []
    all_wrong = []

    for model in model_names:
        correct = [
            yt == yp
            for yt, yp in zip(y_true, predictions[model])
        ]
        correct_matrix.append(correct)
    correct_matrix = list(zip(*correct_matrix))

    for idx, row in enumerate(correct_matrix):
        if all(row):
            all_correct.append(idx)
        elif not any(row):
            all_wrong.append(idx)

    correct_class_dist = Counter(y_true.iloc[all_correct])
    wrong_class_dist = Counter(y_true.iloc[all_wrong])

    print("="*80)
    print("ALL MODELS CORRECT")
    print("="*80)

    print(f"\nCount: {len(all_correct)}")
    print("\nClass distribution:")
    print(
        pd.Series(correct_class_dist)
        .sort_values(ascending=False)
    )

    print("\nExamples:")
    for idx in all_correct[:k_examples]:
        print("\n" + "-"*40)
        print(f"ID: {idx}")
        print(f"TRUE: {y_true.iloc[idx]}")
        print(f"TEXT:\n{X.iloc[idx][:500]}")
    print("\n\n")

    print("="*80)
    print("ALL MODELS WRONG")
    print("="*80)

    print(f"\nCount: {len(all_wrong)}")
    print("\nClass distribution:")
    print(
        pd.Series(wrong_class_dist)
        .sort_values(ascending=False)
    )

    print("\nExamples:")
    for idx in all_wrong[:k_examples]:
        print("\n" + "-"*40)
        print(f"ID: {idx}")
        print(f"TRUE: {y_true.iloc[idx]}")
        print(f"TEXT:\n{X.iloc[idx][:500]}")
        print("\nPredictions:")

        for model in model_names:
            print(
                f"{model}: "
                f"{predictions[model][idx]}"
            )

    return {
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "correct_distribution": correct_class_dist,
        "wrong_distribution": wrong_class_dist
    }