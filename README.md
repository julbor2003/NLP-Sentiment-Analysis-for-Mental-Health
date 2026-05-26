# Sentiment Analysis for Mental Health

## Projekt zaliczeniowy — **NLP 2026**

### Autor: *Julia Borowska*

---

**Uwaga:** W repozytorium tym znajdują się treści dotyczące zdrowia psychicznego, depresji, kryzysów emocjonalnych oraz zachowań suicydalnych.

---

## Źródło danych

Dataset wykorzystany w projekcie pochodzi z platformy Kaggle:

`https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health`

Zbiór został wybrany ze względu na:
- tematykę związaną z analizą zdrowia psychicznego,
- dużą liczbę przykładów tekstowych,
- wieloklasowy charakter problemu,
- obecność rzeczywistych danych pochodzących z mediów społecznościowych.

Dataset stanowi połączenie kilku publicznie dostępnych zbiorów dotyczących zdrowia psychicznego, m.in.:
- Reddit Mental Health Data,
- Depression Reddit Cleaned,
- Suicidal Tweet Detection Dataset,
- Human Stress Prediction.

Dane pochodzą głównie z:
- Reddita,
- Twittera/X,
- forów internetowych,
- postów i komentarzy w mediach społecznościowych.

Każdy analizowany wpis zawiera:
- identyfikator (`unique_id`),
- treść wypowiedzi (`statement`),
- przypisaną klasę (`status`).

---

### Klasy w datasetcie

W zbiorze tym mamy **siedem klas opisujących stan emocjonalny lub rodzaj problemu psychicznego** obecnego w wypowiedzi:

| Klasa | Opis |
|---|---|
| `Normal` | neutralne wypowiedzi niezawierające wyraźnych sygnałów problemów psychicznych |
| `Depression` | wypowiedzi związane z depresją, poczuciem beznadziei, obniżonym nastrojem |
| `Suicidal` | treści zawierające odniesienia do myśli samobójczych lub self-harmu |
| `Anxiety` | wypowiedzi związane z lękiem, paniką lub chronicznym niepokojem |
| `Stress` | teksty opisujące stres i przeciążenie emocjonalne |
| `Bipolar` | wypowiedzi związane z chorobą afektywną dwubiegunową |
| `Personality disorder` | treści odnoszące się do zaburzeń osobowości |

Warto podkreślić, że granice pomiędzy poszczególnymi klasami nie są ostre. Wiele wypowiedzi jednocześnie zawiera sygnały depresji, lęku, stresu czy kryzysu suicydalnego, dlatego nawet dla człowieka jednoznaczna klasyfikacja nie zawsze jest możliwa.

Problem klasyfikacji w tym datasetcie ma więc charakter **silnie semantyczny i częściowo subiektywny**. Część błędów modeli może zatem wynikać nie tylko z ograniczeń algorytmów, ale również z **naturalnego nakładania się stanów emocjonalnych oraz niejednoznaczności samych etykiet**.

---

### Czyszczenie danych

Po usunięciu wybrakowanych rekordów dataset zawierał: **52 681 przykładów** (`362` rekordy zostały odrzucone z powodu brakujących danych tekstowych).

Podstawowy preprocessing wykorzystywany w większości eksperymentów obejmował:
- zamianę tekstu na małe litery,
- usunięcie nadmiarowych białych znaków.

Dodatkowo testowano eksperymentalne warianty preprocessingu:
* usuwanie stopwords,
* pozostawienie wyłącznie tokenów alfanumerycznych,
* lematyzację (`WordNetLemmatizer`),
* lematyzację uwzględniającą części mowy (`POS-aware lemmatization`).

---

### Podział danych

Dataset został podzielony **w sposób stratyfikowany** na:
* zbiór treningowy — `70%` (36 876 przykładów),
* zbiór walidacyjny — `15%` (7 902 przykładów),
* zbiór testowy — `15%` (7 903 przykładów).

**Więcej informacji o zbiorze danych w notebooku: `data.ipynb`.**
 
---

## Przeprowadzone eksperymenty

1. **Czy oczyszczanie tekstu poprawia wyniki?**

    - **Preprocessing v1**:
        - usunięcie stopwords 
        - lematyzacja `WordNetLemmatizer`
    - **Preprocessing v2**:
        - usunięcie stopwords
        - tylko alfanumeryczne tokeny
        - lematyzacja `WordNetLemmatizer`
    - **Preprocessing v3**:
        - usunięcie stopwords 
        - lematyzacja uwzględniająca części mowy
    - **Preprocessing v4**:
        - usunięcie stopwords 
        - tylko alfanumeryczne tokeny
        - lematyzacja uwzględniająca części mowy
    - **Badanie zależności pomiędzy wartością parametru `max_features` w `TfidfVectorizer` a otrzymywanym `f1_score`.**

---

2. **Co daje użycie mocniejszego modelu (SVM) i poszerzenie kontekstu (bigramy, trigramy)?**

    - **2.1.** Zastąpienie `LogisticRegression` modelem `LinearSVC` i ponowne zbadanie zależności pomiędzy wartością parametru `max_features` w `TfidfVectorizer` a otrzymywanym `f1_score`.
    - **2.2.** Szerszy kontekst uzyskany poprzez TF-IDF z użyciem **bigramów** (zarówno dla `LogisticRegression` jak i `LinearSVC`).
    - **2.3.** Szerszy kontekst uzyskany poprzez TF-IDF z użyciem **trigramów** (zarówno dla `LogisticRegression` jak i `LinearSVC`).

---

3. **Czy model polega na długości klasyfikowanego tekstu?**
    - Jak wyniki klasyfikacj się zmieniają, po skróconieu klasyfikowanego tekstu do **max. 100 znaków** (zarówno przy użcyciu modelu `LogisticRegression` jak i `LinearSVC`)?
    - Jak zmienia się jakości klasyfikatorów w zależności od max. długości tekstu (od 50 do 1000 znaków)?
    - Jak przycięcie do max. 100 znaków odbija się na wynikach dla każdej z klasyfikowanych klas? **Czy widzimy korelację spadku jakości ze średnią długością tekstu w klasie?**

---

4. **Klasyfikacja binarna, rozpoznanie problemu i klasyfikacja hierarchiczna.**
    - Wszystkie warianty tego eksperymentu wykonuję na trzech różnych modelach:
        - `LogisticRegression`
        - `LinearSVC`
        - `LinearSVC` w oparciu o TF-IDF z użyciem **bigramów**
    - **5.1.** Klasyfikacja binarna: teksty wskazujące na stabilny stan psychiczny (klasa `Normal`) vs. pozostałe klasy (związane z trapiącym autora/autorkę problemem). **Jak dłogość tekstu wpływa na ten wariant klasyfikacji?**
    - **5.2.** Klasyfikacja ograniczona do tekstów pochodzących z klas powiązanych z problememi emocjonalnymi/psychicznymi (po wykluczeniu klasy `Normal`).
    - **5.3.** Czy przeprowadzenie klasyfikacji etapami (pierwsze rozpoznanie czy problem występuje, a następnei ewentualna jego precyzyjna identyfikacja) poprawia jakość modelu? 

---

## Wyniki

**Pełny opis wyników znajduje się w pliku:** `results.md`

---

### Analiza błędów

**Pełna analiza błędów znajduje się w pliku:** `errors.md`

---

## Wnioski

Projekt pokazał, że klasyfikacja tekstów związanych ze zdrowiem psychicznym jest zadaniem trudnym nie tylko ze względu na użycie prostych modeli NLP, ale również na sam charakter danych. Granice pomiędzy klasami takimi jak `Depression`, `Suicidal`, `Stress` czy `Anxiety` często są bardzo płynne, a część błędów modeli było semantycznie uzasadnionych nawet wtedy, gdy klasyfikacja formalnie była niepoprawna.

**Największą poprawę jakości przyniosło:**
- zastosowanie mocniejszego klasyfikatora: `LinearSVC`,
- wykorzystanie bigramów w reprezentacji TF-IDF,
- pozostawienie stosunkowo prostego preprocessingu.

Istotne okazało się również zachowanie większej ilości oryginalnej formy tekstu. W danych pochodzących z social mediów styl wypowiedzi, powtórzenia, interpunkcja czy chaotyczny zapis często niosły dodatkową informację emocjonalną.

**Nieskuteczne okazały się:**
- agresywny preprocessing,
- lematyzacja zwracająca uwagę na części mowy,
- rozszerzanie kontekstu bez użycia mocniejszego klasyfikatora.

**Najbardziej zaskakujące okazało się, że:**
- bardzo prosty baseline (`TF-IDF + Logistic Regression`) osiągał stosunkowo dobre wyniki,
- mocniejszy preprocessing często pogarszał rezultaty,
- wiele błędów modeli było „rozsądnych” semantycznie.

**Projekt pozwolił lepiej zrozumieć:**
- działanie klasycznych metod NLP opartych o TF-IDF,
- wpływ preprocessingu na reprezentację tekstu,
- różnice między `LogisticRegression` i `LinearSVC`,
- znaczenie doboru reprezentacji (`unigramy`, `bigramy`, `trigramy`),
- wpływ jakości i charakteru danych na końcowe wyniki modeli.

Najważniejszy wniosek jest jednak taki, że w zadaniach związanych ze zdrowiem psychicznym ograniczeniem bardzo często nie jest sam model, lecz niejednoznaczność języka i samych etykiet. **Nawet człowiek nie zawsze byłby w stanie jednoznacznie przypisać część wypowiedzi do pojedynczej kategorii.**

---

### Ograniczenia datasetu

Dataset posiada kilka istotnych ograniczeń wpływających na jakość klasyfikacji:
* silne nakładanie się semantyczne części klas (`Depression`, `Suicidal`, `Stress`, `Anxiety`),
* niezbalansowany rozkład klas,
* obecność szumu charakterystycznego dla social mediów,
* liczne błędy językowe, slang i niestandardowa interpunkcja,
* **bardzo zróżnicowana długość tekstów**,
* możliwe niejednoznaczności etykiet.

Analiza błędów sugeruje również, że część etykiet mogła być przypisywana częściowo na podstawie źródła danych (np. subredditów), a nie wyłącznie samej treści wypowiedzi.