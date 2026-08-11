# Porównanie narzędzi do ewaluacji systemów RAG

## Cel

Celem analizy jest porównanie narzędzi **RAGAS**, **DeepEval**,
**TruLens** i **Arize Phoenix** pod kątem wykorzystania w prototypie RAG
opartym o **PostgreSQL + pgvector**.

Porównanie obejmuje:

-   łatwość integracji z pgvector,
-   wspierane metryki RAG,
-   koszt ewaluacji i wykorzystanie LLM-as-a-Judge,
-   aktywność projektu,
-   jakość dokumentacji,
-   przydatność do dalszych eksperymentów.

> **Ważne:** pgvector odpowiada za przechowywanie i wyszukiwanie
> wektorów. Framework ewaluacyjny nie musi mieć dedykowanego adaptera
> pgvector. Wystarczy, że aplikacja przekaże mu pytanie, pobrane
> fragmenty, odpowiedź modelu oraz --- jeśli dana metryka tego wymaga
> --- dane referencyjne z golden datasetu.

------------------------------------------------------------------------

## Kryteria porównania

### Integracja z pgvector

Typowy przepływ danych wygląda następująco:

``` text
query
  ↓
embedding
  ↓
PostgreSQL + pgvector
  ↓
retrieved chunks
  ↓
LLM
  ↓
answer
  ↓
framework ewaluacyjny
```

Z tego powodu RAGAS i DeepEval można wykorzystać niezależnie od
konkretnego vector store. Wyniki wyszukiwania z pgvector są po prostu
przekazywane do frameworka jako `retrieved_contexts` /
`retrieval_context`.

TruLens i Phoenix są bardziej nastawione również na instrumentację i
obserwowalność całego pipeline'u, dlatego ich integracja może obejmować
dodatkowo tracing operacji retrieval i generation.

------------------------------------------------------------------------

## Tabela porównawcza

| Kryterium | RAGAS | DeepEval | TruLens | Arize Phoenix |
| --- | --- | --- | --- | --- |
| Główne zastosowanie | Ewaluacja RAG/LLM | Testowanie i ewaluacja LLM/RAG | Ewaluacja + tracing | Observability + tracing + evals |
| Integracja z pgvector | **Łatwa** — niezależny od vector store | **Łatwa** — niezależny od vector store | **Dobra** — możliwość instrumentacji pipeline'u | **Dobra**, ale wymaga więcej konfiguracji |
| Faithfulness / groundedness | Tak | Tak | Tak | Tak |
| Answer relevance | Tak | Tak | Tak | Tak |
| Context / document relevance | Tak | Tak | Tak | Tak |
| Context precision / recall | Tak | Tak | Możliwe przez metryki/feedback | Możliwe przez evaluatory |
| Klasyczne Precision@k / Recall@k / MRR / nDCG | Najlepiej policzyć samodzielnie | Najlepiej policzyć samodzielnie | Możliwe jako własne metryki | Możliwe jako code-based evaluators |
| Custom metrics | Tak | Tak | Tak | Tak |
| LLM-as-a-Judge | Tak | Tak | Tak | Tak |
| Metryki bez dodatkowego LLM | Tak | Tak, głównie własne/custom | Tak | Tak — code-based evals |
| Golden datasets / eksperymenty | Tak | Tak | Tak | **Bardzo dobre wsparcie** |
| Pytest / testy regresyjne | Możliwe | **Bardzo dobre** | Możliwe | Dostępne mechanizmy eksperymentów/testów |
| Tracing / observability | Ograniczone | Dostępne | **Bardzo dobre** | **Bardzo dobre** |
| Złożoność wdrożenia | **Niska** | **Niska/średnia** | Średnia | Średnia/wyższa |
| Dokumentacja | **Bardzo dobra** | **Bardzo dobra** | Dobra / bardzo dobra | **Bardzo dobra** |
| Aktywność projektu | Wysoka | Wysoka | Wysoka | **Bardzo wysoka** |
| Dopasowanie do prostego prototypu ewaluacyjnego | **Bardzo dobre** | **Bardzo dobre** | Dobre | Dobre, ale bardziej rozbudowane niż potrzeba |

------------------------------------------------------------------------

# 1. RAGAS

RAGAS jest frameworkiem skoncentrowanym bezpośrednio na **ewaluacji
aplikacji RAG i LLM**. Dzięki temu jego model pracy dobrze odpowiada
eksperymentowi, w którym chcemy uruchamiać ten sam golden dataset na
różnych konfiguracjach retrievera i generatora.

### Metryki

RAGAS udostępnia m.in.:

-   Context Precision,
-   Context Recall,
-   Context Entities Recall,
-   Faithfulness,
-   Response Relevancy,
-   Noise Sensitivity,
-   Factual Correctness,
-   Semantic Similarity,
-   metryki non-LLM.

W przypadku klasycznych metryk Information Retrieval, takich jak:

-   Precision@k,
-   Recall@k,
-   MRR,
-   nDCG@k,

najlepszym rozwiązaniem jest policzenie ich deterministycznie na
podstawie relevance labels zapisanych w golden datasecie.

### Integracja z pgvector

Integracja jest prosta, ponieważ RAGAS nie musi komunikować się
bezpośrednio z PostgreSQL.

``` text
pgvector
   ↓
retrieved chunks
   ↓
RAGAS evaluation sample
```

Aplikacja wykonuje retrieval, a następnie przekazuje wynik do RAGAS.

### Koszt

Sam framework jest open-source. Koszt może pojawić się podczas używania
metryk opartych o LLM lub zewnętrzny model embeddingowy.

Przykładowo:

-   klasyczne Precision@k / Recall@k / MRR / nDCG → **bez LLM**,
-   Faithfulness → **LLM-as-a-Judge**,
-   Response Relevancy → **LLM + embeddings**,
-   część wariantów Context Precision/Recall → może używać LLM,
-   istnieją również warianty non-LLM / ID-based.

Koszt zależy więc od wybranego zestawu metryk i modeli, a nie od samego
użycia RAGAS.

### Zalety

-   skoncentrowany na RAG evaluation,
-   niski próg wejścia,
-   metryki dobrze dopasowane do retrieval + generation,
-   możliwość łączenia metryk LLM i non-LLM,
-   łatwe użycie niezależnie od vector store.

### Wady

-   nie jest pełną platformą observability,
-   część metryk semantycznych generuje dodatkowe wywołania LLM,
-   klasyczne metryki rankingowe nadal warto implementować osobno.

------------------------------------------------------------------------

# 2. DeepEval

DeepEval jest frameworkiem do ewaluacji aplikacji LLM zaprojektowanym w
sposób przypominający klasyczne testowanie oprogramowania.

Duży nacisk kładzie na:

``` text
test case
   ↓
metric
   ↓
threshold
   ↓
pass / fail
```

oraz integrację z `pytest`.

### Metryki

Dla RAG dostępne są m.in.:

-   Faithfulness,
-   Answer Relevancy,
-   Contextual Precision,
-   Contextual Recall,
-   Contextual Relevancy.

DeepEval oferuje obecnie ponad 50 gotowych metryk obejmujących również
agentów, bezpieczeństwo, konwersacje i inne zastosowania LLM.

### Integracja z pgvector

Podobnie jak RAGAS, DeepEval nie wymaga bezpośredniej integracji z
pgvector.

``` text
PostgreSQL + pgvector
        ↓
retrieval_context
        ↓
LLMTestCase
        ↓
DeepEval metrics
```

Dzięki temu można zachować własną implementację retrievalu i użyć
DeepEval wyłącznie jako warstwy ewaluacyjnej.

### Koszt

Dokumentacja DeepEval wskazuje, że **większość gotowych metryk
wykorzystuje LLM-as-a-Judge**.

Oznacza to, że np.:

-   Faithfulness,
-   Answer Relevancy,
-   Contextual Precision,
-   Contextual Recall

mogą powodować dodatkowe wywołania modelu oceniającego.

Klasyczne Precision@k, Recall@k, MRR i nDCG nadal można liczyć
samodzielnie bez kosztu LLM.

### Zalety

-   bardzo dobra integracja z `pytest`,
-   możliwość ustawiania progów `pass/fail`,
-   dobre wsparcie testów regresyjnych,
-   duża liczba gotowych metryk,
-   możliwość debugowania wyników LLM-as-a-Judge,
-   lokalne wykonywanie ewaluacji.

### Wady

-   większość gotowych metryk korzysta z LLM-as-a-Judge,
-   dla prostego eksperymentu część możliwości frameworka może być
    zbędna,
-   klasyczne metryki IR nadal lepiej liczyć deterministycznie.

------------------------------------------------------------------------

# 3. TruLens

TruLens łączy **ewaluację z tracingiem i observability**.

W kontekście RAG szczególnie istotna jest koncepcja oceny relacji:

``` text
Question
   ↓
Context Relevance
   ↓
Retrieved Context
   ↓
Groundedness
   ↓
Answer
   ↓
Answer Relevance
```

Pozwala to analizować nie tylko końcowy wynik eksperymentu, ale również
poszczególne elementy pipeline'u.

### Integracja z pgvector

TruLens może działać niezależnie od vector store, jeżeli retrieval
zostanie odpowiednio zinstrumentowany. Projekt rozwija również wsparcie
dla PostgreSQL jako elementu swojej infrastruktury.

W porównaniu z RAGAS/DeepEval integracja jest jednak bardziej związana z
obserwowaniem wykonania aplikacji niż tylko przekazaniem gotowego
zestawu danych do ewaluacji.

### Koszt

TruLens obsługuje zarówno:

-   metryki deterministyczne/custom,
-   modele lokalne,
-   LLM-as-a-Judge.

Koszt zależy więc od providera użytego przez evaluator.

### Zalety

-   bardzo dobre tracing i observability,
-   możliwość diagnozowania konkretnych requestów,
-   dobre podejście do oceny całego pipeline'u,
-   aktywnie rozwijany projekt.

### Wady

-   większa złożoność niż potrzebna do prostego benchmarku,
-   wymaga więcej instrumentacji aplikacji,
-   mniej bezpośredni niż RAGAS przy prostym eksperymencie offline.

------------------------------------------------------------------------

# 4. Arize Phoenix

Phoenix jest najbardziej rozbudowanym z analizowanych rozwiązań pod
względem **AI observability**.

Łączy:

-   tracing,
-   ewaluację,
-   datasety,
-   eksperymenty,
-   analizę retrievalu,
-   analizę kosztu i tokenów,
-   UI do badania poszczególnych wykonań.

Phoenix wykorzystuje OpenTelemetry i OpenInference do instrumentowania
aplikacji.

### Integracja z pgvector

Phoenix nie wymaga zastąpienia pgvector własnym vector store. Retrieval
może pozostać częścią aplikacji, a operacja wyszukiwania może zostać
zapisana jako span/trace.

``` text
query
  ↓
pgvector retrieval ─── trace/span
  ↓
contexts ───────────── evaluation
  ↓
LLM ────────────────── trace/span
  ↓
answer ─────────────── evaluation
```

### Koszt

Phoenix obsługuje:

-   LLM-based evaluators,
-   code-based checks,
-   human labels.

Sam open-source Phoenix można self-hostować. Koszt LLM pojawia się
wtedy, gdy używane są evaluatory oparte o zewnętrzny model.

### Zalety

-   bardzo dobre observability,
-   tracing retrievalu i generation,
-   datasety i eksperymenty,
-   UI ułatwiające analizę błędów,
-   bardzo aktywny rozwój projektu.

### Wady

-   większa złożoność infrastrukturalna,
-   do pierwszego eksperymentu RAG oferuje więcej funkcji niż jest
    konieczne,
-   wymaga instrumentacji, jeśli chcemy wykorzystać pełnię możliwości.

------------------------------------------------------------------------

# Koszt ewaluacji -- najważniejsze rozróżnienie

Koszt należy rozpatrywać na dwóch poziomach.

## Metryki deterministyczne

Przy poprawnie przygotowanym golden datasecie:

``` text
Precision@k
Recall@k
MRR
nDCG@k
```

mogą zostać policzone bez jakiegokolwiek wywołania LLM.

Potrzebne są jedynie:

``` text
expected relevant chunks/documents
            ↕
actual retrieved chunks/documents
```

Koszt ewaluacji jest wtedy pomijalny.

## LLM-as-a-Judge

Metryki takie jak:

``` text
Faithfulness
Answer Relevance
Context Relevance
```

często wymagają dodatkowego modelu oceniającego.

Przykładowy pipeline:

``` text
RAG wygenerował odpowiedź
        ↓
evaluator LLM
        ↓
ocena odpowiedzi
```

Oznacza to, że jeden request do systemu RAG może spowodować kilka
dodatkowych wywołań LLM podczas ewaluacji.

Przy korzystaniu z płatnego API koszt rośnie wraz z:

-   liczbą pytań w golden datasecie,
-   liczbą ocenianych konfiguracji,
-   liczbą metryk LLM-based,
-   liczbą wywołań potrzebnych do policzenia jednej metryki,
-   długością kontekstu.

Dlatego klasyczne metryki retrieval powinny pozostać deterministyczne.

------------------------------------------------------------------------

# Aktywność projektów

Wszystkie cztery projekty są aktywnie rozwijane.

Na moment przygotowania analizy:

-   **TruLens** posiada aktualne wydania z lipca 2026 (m.in. 2.10.0) i
    rozwija nowe API metryk, tracing oraz mechanizmy ewaluacji.
-   **Arize Phoenix** jest rozwijany bardzo intensywnie; seria 19.x
    otrzymywała liczne wydania w lipcu 2026, obejmujące m.in.
    eksperymenty, ewaluacje, tracing i UI.
-   **RAGAS** posiada rozwijaną dokumentację aktualnego zestawu metryk
    RAG i mechanizm tworzenia własnych metryk.
-   **DeepEval** pozostaje aktywnym frameworkiem o szerokim zakresie
    metryk i rozbudowanym ekosystemie testowania LLM.

Sama częstotliwość commitów/releases nie powinna jednak decydować o
wyborze. Ważniejsze jest dopasowanie architektury narzędzia do celu
prototypu.

------------------------------------------------------------------------

# Rekomendacja

Do dalszej pracy rekomendowane są:

## 1. RAGAS -- główny wybór

RAGAS najlepiej odpowiada obecnemu celowi: **kontrolowanej ewaluacji
eksperymentalnego systemu RAG**.

Proponowany podział odpowiedzialności:

``` text
                 Evaluation pipeline

Precision@k ─┐
Recall@k    ─┤
MRR         ─┼── własne metryki deterministyczne
nDCG@k      ─┘
                \
                 → raport eksperymentu
                /
Faithfulness ───┐
Answer Relevance┴── RAGAS
```

Pozwala to uniknąć niepotrzebnych kosztów LLM przy klasycznych metrykach
retrieval, a jednocześnie wykorzystać RAGAS tam, gdzie ocena semantyczna
jest rzeczywiście potrzebna.

## 2. DeepEval -- drugi kandydat

DeepEval warto zachować jako drugie narzędzie do dalszego sprawdzenia,
szczególnie jeśli ewaluacja ma później wejść do:

``` text
pytest
  ↓
CI/CD
  ↓
regression tests
```

Jego podejście testowe może być bardzo wartościowe po zakończeniu
początkowej fazy eksperymentalnej.

------------------------------------------------------------------------

# Dlaczego nie TruLens lub Phoenix?

Nie dlatego, że są słabszymi narzędziami.

Ich główną przewagą jest:

``` text
evaluation
+
tracing
+
observability
```

W pierwszym prototypie głównym celem jest natomiast:

``` text
golden dataset
      ↓
różne konfiguracje RAG
      ↓
metryki
      ↓
porównanie wyników
```

RAGAS i DeepEval rozwiązują ten problem przy mniejszej liczbie
dodatkowych komponentów.

Jeżeli projekt zostanie później rozszerzony o monitoring działającego
systemu RAG, analizę produkcyjnych requestów i tracing, wtedy **Phoenix
lub TruLens mogą stać się lepszym uzupełnieniem stosu**.

------------------------------------------------------------------------

# Podsumowanie

Dla prototypu wykorzystującego **PostgreSQL + pgvector** wybór vector
store nie stanowi istotnego ograniczenia dla frameworka ewaluacyjnego.
Retrieval może pozostać całkowicie własną implementacją, a framework
otrzymywać jego wyniki jako dane wejściowe.

Najbardziej odpowiednim rozwiązaniem na obecnym etapie jest:

1.  **RAGAS** -- podstawowe narzędzie do ewaluacji semantycznej RAG.
2.  **DeepEval** -- alternatywa / drugie narzędzie, szczególnie
    interesujące pod kątem testów regresyjnych i CI/CD.
3.  **Własne metryki deterministyczne** -- Precision@k, Recall@k, MRR i
    nDCG@k.
4.  **LLM-as-a-Judge tylko tam, gdzie jest potrzebny** -- przede
    wszystkim Faithfulness i Answer Relevance.

Takie podejście zapewnia prosty pipeline, ogranicza koszt ewaluacji i
pozwala niezależnie mierzyć jakość retrievalu oraz generowanej
odpowiedzi.

## Źródła

-   RAGAS Documentation -- Available Metrics:
    https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
-   DeepEval Documentation -- Introduction:
    https://deepeval.com/docs/introduction
-   DeepEval Documentation -- Metrics:
    https://deepeval.com/docs/metrics-introduction
-   TruLens Documentation: https://www.trulens.org/
-   TruLens GitHub Releases: https://github.com/truera/trulens/releases
-   Arize Phoenix Documentation: https://arize.com/docs/phoenix
-   Arize Phoenix GitHub Releases:
    https://github.com/Arize-ai/phoenix/releases
