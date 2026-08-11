# Ewaluacja systemów RAG — metryki, golden dataset i pułapki

## 1. Cel ewaluacji RAG

Ewaluację systemu RAG warto rozdzielić na dwa niezależne obszary:

1. **Retrieval evaluation** — ocena, czy retriever znajduje właściwe fragmenty dokumentów i umieszcza je odpowiednio wysoko w rankingu.
2. **Generation evaluation** — ocena, czy odpowiedź modelu jest zgodna z pobranym kontekstem i odpowiada na pytanie użytkownika.

Takie rozdzielenie pozwala ustalić źródło błędu. Niepoprawna odpowiedź może wynikać zarówno z niewłaściwie znalezionego kontekstu, jak i z błędu modelu generującego odpowiedź.

---

## 2. Metryki retrieval

### Precision@k

**Precision@k** określa, jaka część pierwszych `k` wyników zwróconych przez retriever jest relewantna dla danego pytania.

**Wzór:**

`Precision@k = liczba relewantnych wyników w top-k / k`

Przykład: jeśli spośród 5 zwróconych chunków 3 są relewantne, `Precision@5 = 3/5 = 0.6`.

Wysoki precision oznacza, że do LLM trafia mało zbędnego kontekstu. Jest to istotne, ponieważ nieistotne fragmenty mogą zwiększać koszt, zajmować context window i utrudniać modelowi wygenerowanie poprawnej odpowiedzi.

### Recall@k

**Recall@k** określa, jaka część wszystkich znanych relewantnych wyników została odnaleziona w pierwszych `k` wynikach.

**Wzór:**

`Recall@k = liczba znalezionych relewantnych wyników / liczba wszystkich relewantnych wyników`

Jeśli dla pytania istnieją 4 relewantne fragmenty, a retriever znalazł 3 z nich w top-5, `Recall@5 = 3/4 = 0.75`.

Recall jest szczególnie ważny, gdy kompletna odpowiedź wymaga informacji rozproszonych między kilkoma fragmentami dokumentacji.

### MRR — Mean Reciprocal Rank

**MRR** mierzy, jak wysoko w rankingu pojawia się **pierwszy relewantny wynik**.

Dla pojedynczego pytania obliczany jest Reciprocal Rank:

`RR = 1 / pozycja pierwszego relewantnego wyniku`

Jeżeli pierwszy poprawny wynik znajduje się na pozycji 1, RR wynosi `1.0`; na pozycji 2 — `0.5`; na pozycji 4 — `0.25`. MRR jest średnią RR dla wszystkich pytań testowych.

Metryka jest przydatna, gdy oczekujemy, że przynajmniej jeden bardzo dobry fragment powinien znaleźć się jak najwyżej.

### nDCG@k — Normalized Discounted Cumulative Gain

**nDCG@k** ocenia jakość całego rankingu i pozwala uwzględnić **różne poziomy relewancji** wyników, np.:

- `0` — irrelevant,
- `1` — partially relevant,
- `2` — relevant,
- `3` — highly relevant.

Metryka premiuje sytuację, w której najbardziej relewantne wyniki znajdują się na początku rankingu. Wynik jest normalizowany względem idealnego uporządkowania i mieści się zwykle w zakresie `0–1`, gdzie `1` oznacza idealny ranking.

nDCG jest bardziej informacyjne niż proste metryki binarne, jeśli nie wszystkie poprawne fragmenty mają taką samą wartość dla odpowiedzi.

---

## 3. Metryki generowania odpowiedzi

### Faithfulness

**Faithfulness** ocenia, czy twierdzenia zawarte w wygenerowanej odpowiedzi są poparte kontekstem dostarczonym modelowi przez retriever.

Metryka odpowiada na pytanie:

> Czy model odpowiada na podstawie znalezionych dokumentów, czy dodaje informacje, których w kontekście nie ma?

Jest to jedna z podstawowych metod wykrywania halucynacji w RAG. W praktyce faithfulness jest często oceniane metodą **LLM-as-a-Judge**: dodatkowy model analizuje twierdzenia z odpowiedzi i sprawdza, czy wynikają one z retrieved context.

Faithfulness nie wymaga wzorcowej odpowiedzi, jeżeli oceniana jest bezpośrednio relacja `answer ↔ retrieved context`.

### Answer Relevance

**Answer Relevance** ocenia, czy wygenerowana odpowiedź faktycznie odpowiada na pytanie użytkownika.

Odpowiedź może być poprawna faktograficznie i w pełni oparta na kontekście, ale jednocześnie nie odpowiadać na zadane pytanie. Dlatego answer relevance mierzy inny aspekt niż faithfulness:

- **Faithfulness:** `answer ↔ context`
- **Answer Relevance:** `answer ↔ question`

Ta metryka również jest często realizowana za pomocą LLM-as-a-Judge lub metod opartych na podobieństwie semantycznym.

---

## 4. Ground truth i golden dataset

**Ground truth** to wzorcowa informacja określająca, co dla danego przypadku testowego uznajemy za poprawne. Dla retrievalu mogą to być identyfikatory lub zakresy relewantnych fragmentów dokumentów, a dla generowania — referencyjna odpowiedź.

**Golden dataset** jest zbiorem wielu przypadków testowych wraz z ich ground truth.

Przykładowa struktura przypadku:

```json
{
  "question": "How do I delete a collection?",
  "expected_answer": "...",
  "relevant_evidence": [
    {
      "document_id": "collections.md",
      "start": 1200,
      "end": 1480,
      "text": "..."
    }
  ]
}
```

Przechowywanie dowodu jako **dokument + pozycja w źródle + tekst** jest bardziej odporne na zmiany chunkingu niż zapisanie wyłącznie `chunk_id`. Przy zmianie wielkości chunków lub overlapu identyfikatory chunków mogą się zmienić, podczas gdy źródłowy fragment dokumentu pozostaje tym samym ground truth.

---

## 5. Metodologie budowania golden datasetów

### Podejście manualne

Ekspert lub osoba przygotowująca benchmark analizuje dokumenty, tworzy pytania i ręcznie oznacza poprawne odpowiedzi oraz relewantne fragmenty.

**Zalety:** wysoka kontrola jakości, możliwość tworzenia realistycznych i trudnych przypadków.

**Wady:** duży koszt czasowy i ograniczona skalowalność.

### Podejście syntetyczne

LLM otrzymuje dokumenty lub ich fragmenty i generuje na ich podstawie pytania, odpowiedzi oraz evidence.

**Zalety:** szybkość i możliwość wygenerowania dużego datasetu.

**Wady:** ryzyko sztucznie łatwych pytań, błędów modelu oraz zbyt dużego podobieństwa pytań do tekstu źródłowego.

### Podejście hybrydowe

LLM generuje kandydatów, a człowiek je weryfikuje, poprawia i odrzuca przypadki niskiej jakości.

Jest to praktyczny kompromis dla prototypów: automatyzuje najbardziej czasochłonną część pracy, zachowując kontrolę jakości benchmarku.

### Dane z rzeczywistych zapytań

W istniejącym systemie można wykorzystać anonimowe, rzeczywiste pytania użytkowników, a następnie ręcznie przygotować dla nich ground truth. Takie dane dobrze odwzorowują rzeczywisty rozkład zapytań, ale wymagają działającego systemu i odpowiednich danych historycznych.

---

## 6. Najczęstsze pułapki przy ewaluacji

### Data leakage

Nie należy optymalizować konfiguracji RAG i jednocześnie raportować końcowej jakości na dokładnie tych samych przypadkach. Warto rozdzielić dane wykorzystywane do strojenia od finalnego zestawu testowego.

### Synthetic bias

Pytania generowane przez LLM bezpośrednio z pojedynczego chunka bywają zbyt podobne semantycznie do źródła. Retriever może osiągać bardzo dobre wyniki na takim benchmarku, mimo że gorzej radzi sobie z naturalnymi pytaniami użytkowników.

### Niepełne relevance labels

Ta sama informacja może znajdować się w kilku miejscach dokumentacji. Jeśli golden dataset oznacza tylko jeden z poprawnych fragmentów, pozostałe poprawne wyniki retrievera zostaną błędnie uznane za irrelevant.

### Zależność ground truth od chunkingu

Jeżeli ground truth jest zapisany wyłącznie jako `chunk_id`, zmiana strategii chunkowania może unieważnić dataset. Lepiej oznaczać evidence w oryginalnym dokumencie i mapować je na chunki dla konkretnej konfiguracji eksperymentu.

### Ocena wyłącznie średnich wyników

Średni `Recall@5 = 0.9` może ukrywać grupę pytań, dla których system działa bardzo źle. Oprócz wyników zagregowanych należy analizować wyniki per-query oraz — jeśli dataset to umożliwia — według typów pytań.

### Brak rozdzielenia retrievalu i generation

Ocena wyłącznie finalnej odpowiedzi utrudnia diagnozę błędów. Słaba odpowiedź może wynikać z retrievera, generatora albo obu komponentów. Retrieval i generation powinny być mierzone osobno.

### Bezrefleksyjne używanie LLM-as-a-Judge

Oceny LLM nie są całkowicie deterministycznym ground truth. Wynik może zależeć od modelu, promptu i konfiguracji. Dodatkowo każda taka ewaluacja może generować koszt wywołań API. Klasyczne metryki IR, takie jak Precision@k, Recall@k, MRR i nDCG, powinny być liczone deterministycznie, jeśli dostępne są odpowiednie relevance labels.

---

## 7. Podsumowanie metryk

| Metryka | Warstwa | Co mierzy | Typowy ground truth | Wymaga LLM do obliczenia? |
| --- | --- | --- | --- | --- |
| Precision@k | Retrieval | Odsetek relewantnych wyników w top-k | relevance labels | Nie |
| Recall@k | Retrieval | Pokrycie wszystkich relewantnych wyników | relevance labels | Nie |
| MRR | Retrieval | Pozycję pierwszego relewantnego wyniku | relevance labels | Nie |
| nDCG@k | Retrieval | Jakość i kolejność całego rankingu | relevance grades | Nie |
| Faithfulness | Generation | Czy odpowiedź jest poparta kontekstem | retrieved context | Zwykle tak |
| Answer Relevance | Generation | Czy odpowiedź odpowiada na pytanie | pytanie; opcjonalnie reference | Zwykle tak |

## 8. Rekomendowane podejście do prototypu

Dla prototypowego systemu RAG warto zastosować dwuwarstwową ewaluację:

1. **Deterministyczna ewaluacja retrievalu** — Precision@k, Recall@k, MRR i nDCG@k obliczane względem ręcznie zweryfikowanego ground truth.
2. **Semantyczna ewaluacja generowania** — Faithfulness i Answer Relevance, potencjalnie z wykorzystaniem LLM-as-a-Judge.

Golden dataset powinien zawierać realistyczne pytania, referencyjne odpowiedzi oraz evidence wskazane w oryginalnych dokumentach. Dla prototypu dobrym kompromisem jest przygotowanie datasetu metodą hybrydową: wygenerowanie kandydatów automatycznie i następnie ręczna walidacja każdego przypadku.

Takie podejście pozwala niezależnie mierzyć jakość wyszukiwania i generowania oraz diagnozować, który element pipeline'u RAG odpowiada za pogorszenie wyniku.
