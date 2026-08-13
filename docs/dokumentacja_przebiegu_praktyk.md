# Dokumentacja przebiegu praktyk zawodowych

## Cel praktyk

Celem pracy było przygotowanie podstaw prototypu służącego do porównywania jakości retrievalu w systemach RAG. Zakres został podzielony w Jirze na cztery obszary: research, przygotowanie środowiska, budowę pipeline'u ewaluacyjnego oraz podsumowanie wyników.

W ramach dotychczasowych prac:

- przeanalizowałem metody ewaluacji systemów RAG,
- porównałem dostępne narzędzia ewaluacyjne,
- wybrałem publiczny korpus testowy,
- przygotowałem lokalne środowisko z PostgreSQL i pgvector,
- uruchomiłem indeksowanie i wyszukiwanie wektorowe,
- zbudowałem ręcznie zweryfikowany golden dataset.

Implementacja metryk, porównanie konfiguracji, ewaluacja generowania odpowiedzi i dashboard stanowią kolejne etapy projektu.

## Analiza metod i metryk ewaluacji RAG

Pierwszym etapem było zapoznanie się z metodami oceny systemów RAG oraz wybór metryk odpowiednich dla planowanego prototypu. W ramach analizy wyróżniłem dwa obszary, które w przyszłości powinny być oceniane osobno:

1. **Retrieval** — sprawdzenie, czy system znajduje właściwe fragmenty i umieszcza je odpowiednio wysoko w rankingu.
2. **Generation** — sprawdzenie, czy odpowiedź jest zgodna z pobranym kontekstem i odpowiada na pytanie.

Dla planowanej ewaluacji retrievalu wybrałem deterministyczne metryki **Precision@k, Recall@k, MRR i nDCG@k**. Pozwalają one mierzyć odpowiednio czystość wyników, kompletność znalezionego kontekstu, pozycję pierwszego trafienia oraz jakość całego rankingu. Do ewentualnej późniejszej oceny generowania wskazałem **Faithfulness** i **Answer Relevance**, które zwykle wymagają modelu oceniającego. Na tym etapie metryki nie zostały jeszcze zaimplementowane ani uruchomione.

Research wykazał również najważniejsze ryzyka: data leakage, zbyt łatwe pytania syntetyczne, niepełne oznaczenia relewancji oraz uzależnienie ground truth od konkretnego chunkowania. Wnioski te wpłynęły później na sposób przygotowania golden datasetu. Szczegółowy opis znajduje się w `rag_evaluation_research.md`.

## Porównanie narzędzi ewaluacyjnych

Porównałem RAGAS, DeepEval, TruLens i Arize Phoenix pod względem integracji z pgvector, dostępnych metryk, kosztu użycia LLM-as-a-Judge, dokumentacji i złożoności wdrożenia.

Jako głównego kandydata wybrałem **RAGAS**, ponieważ jest bezpośrednio ukierunkowany na ewaluację RAG i może przyjmować wyniki z własnego retrievera. **DeepEval** został wskazany jako druga opcja, szczególnie przydatna w przyszłych testach regresyjnych i integracji z `pytest`. TruLens i Phoenix oferują rozbudowane tracing i observability, ale na etapie prostego prototypu wprowadzają większą złożoność niż jest potrzebna.

Klasyczne metryki retrievalowe mają być liczone samodzielnie, bez wywołań LLM. RAGAS ma być wykorzystywany tam, gdzie potrzebna jest ocena semantyczna. Pełne porównanie znajduje się w `rag_evaluation_tools_comparison.md`.

## Przygotowanie prototypu

Środowisko zostało zbudowane jako zestaw usług uruchamianych przez Docker Compose:

- `backend` odpowiada za wczytywanie chunków, indeksowanie i retrieval,
- `embedding_service` udostępnia model embeddingowy `BAAI/bge-m3`,
- PostgreSQL z rozszerzeniem pgvector przechowuje embeddingi i wykonuje wyszukiwanie wektorowe.

Zależności Pythona są zarządzane przez `uv`, a konfiguracja, między innymi adres bazy, model embeddingowy i ścieżka korpusu, jest przekazywana przez zmienne środowiskowe. Przygotowałem jedną komendę indeksującą cały korpus oraz osobny skrypt do testowego wyszukiwania z konfigurowalnym `top_k`.

Indeksowanie jest idempotentne: ponowne uruchomienie aktualizuje istniejące rekordy zamiast tworzyć duplikaty. Korpus zawiera 735 chunków, a ich identyfikatory są stabilne. Dodałem również testy jednostkowe i integracyjne obejmujące modele danych, klienta embeddingowego, repozytorium, indeksowanie oraz retrieval.

## Wybór danych testowych

Jako publiczne źródło danych wybrałem Open RAGBench (`deepmatics/open_ragbench`) udostępniony na Hugging Face. Wykorzystałem część benchmarku przygotowaną na podstawie publikacji naukowych z arXiv.

W Open RAGBench pliki PDF przetworzono do ustrukturyzowanych dokumentów tekstowych podzielonych na sekcje. Osobno zapisano pytania, odpowiedzi referencyjne, adresy PDF oraz relacje `qrels`, które przypisują pytanie do właściwego dokumentu i sekcji. Benchmark zapewnił więc gotowe pytania i wskazania źródłowe, ale wymagał doprecyzowania fragmentów rzeczywiście potrzebnych do odpowiedzi.

Wybrałem podzbiór o objętości możliwie bliskiej **700 000 znaków**. Limit znaków lepiej kontrolował wielkość korpusu niż sama liczba PDF-ów, ponieważ publikacje znacznie różniły się długością. Ostateczny podzbiór obejmuje:

- 9 dokumentów PDF,
- 698 275 znaków tekstu,
- 50 pytań.

Odrzuciłem pytania wymagające interpretacji obrazu (`text-image` i `text-table-image`), ponieważ prototyp przetwarza tekst. Wybór jest odtwarzalny dzięki manifestowi `selected_documents.json`, a dane pytań i sekcji są zapisane w `selected_questions.json`.

## Przygotowanie golden datasetu

### Problem oznaczania relevant chunks na podstawie Open RAGBench

Open RAGBench zapewnia gotowe pytania oraz relacje `qrels`, które przypisują każdemu pytaniu właściwy dokument i `section_id`. Tekst wskazanej sekcji został zapisany w projekcie jako `ground_truth_text`. Oznacza on obszar dokumentu, na podstawie którego przygotowano pytanie, ale nie wskazuje jeszcze konkretnych chunków, które należy uznać za relevant.

Retriever pracuje na 735 chunkach z `dane.csv`, utworzonych wcześniej z dokumentów PDF. Granice tych chunków nie pokrywają się z granicami sekcji Open RAGBench. Dodatkowo sekcje referencyjne są wyraźnie większe od chunków. Pokazuje to porównanie długości tekstów:

| Rodzaj tekstu | Liczba | Minimum | Mediana | Średnia | Maksimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chunki | 735 | 105 | 1 036 | 976,20 | 2 022 |
| `ground_truth_text` | 50 | 328 | 1 828,50 | 2 824,46 | 13 575 |
| `evidence_texts` | 50 | 76 | 240,50 | 361,40 | 1 199 |

Średni `ground_truth_text` ma około **2,9 razy więcej znaków niż przeciętny chunk**, a jego mediana jest **1,76 raza większa od mediany chunka**. Najdłuższa sekcja ma 13 575 znaków, czyli jest ponad trzynaście razy dłuższa od medianowego chunka. Sekcja referencyjna może więc obejmować wiele chunków.

Analiza treści `ground_truth_text` wykazała również, że znaczna część sekcji nie zawiera informacji istotnych dla odpowiedzi na przypisane pytanie. Gdyby relevant chunks wyznaczać na podstawie samej przynależności do sekcji albo dowolnego pokrycia jej tekstu, za relevant mógłby zostać uznany chunk zawierający wyłącznie poboczny fragment sekcji. Retriever otrzymywałby wtedy pozytywną ocenę za znalezienie tekstu, który nie pozwala odpowiedzieć na pytanie. Powiększałoby to zbiór etykiet relewancji i obniżało jakość golden datasetu, a późniejsze wyniki Precision@k i Recall@k byłyby mało wiarygodne.

### Ekstrakcja evidence z `ground_truth_text`

Aby rozwiązać ten problem, przy pomocy LLM wyekstrahowałem z każdej sekcji tylko fragmenty istotne w kontekście danego pytania. Wynik zapisałem w `evidence_annotations.json` jako listę `evidence_texts`. LLM służył do wskazania fragmentów, a nie do wygenerowania odpowiedzi lub uzupełnienia brakującej wiedzy.

Podczas tworzenia adnotacji obowiązywały następujące wytyczne:

- dla danego pytania można było korzystać wyłącznie z odpowiadającego mu `ground_truth_text`,
- każdy `evidence_text` musiał być dokładnym substringiem tekstu źródłowego,
- zabronione było parafrazowanie, poprawianie tekstu i dodawanie wiedzy zewnętrznej,
- evidence miało obejmować wszystkie informacje potrzebne do odpowiedzi, w tym warunki, ograniczenia i wyjątki,
- jeżeli potrzebne informacje znajdowały się w kilku miejscach sekcji, zapisywano kilka fragmentów,
- fragmenty zachowywano jako pełne zdania, aby nie traciły znaczenia,
- w razie wątpliwości pozostawiano nieco szerszy kontekst, ponieważ pominięcie istotnej informacji było groźniejsze niż niewielki nadmiar,
- usuwano duplikaty oraz informacje całkowicie niezwiązane z pytaniem,
- jeżeli sekcja rzeczywiście nie dostarczałaby przydatnej informacji, lista evidence miała pozostać pusta zamiast zawierać zgadywaną treść.

Adnotacje zostały następnie kilkukrotnie zweryfikowane pod względem kompletności, przydatności, pełności zdań i dokładnej zgodności z tekstem źródłowym. Każdy niepusty fragment został też automatycznie sprawdzony jako substring właściwego `ground_truth_text`. Ostatecznie wszystkie 50 rekordów otrzymało evidence. Mediana evidence wynosi 240,5 znaku, czyli około 23% długości medianowego chunka i około 1/7,6 medianowego `ground_truth_text`.

W dwóch przypadkach pytanie było bardziej szczegółowe od przypisanej sekcji. Zachowano wtedy najlepszy dostępny kontekst z `ground_truth_text`, ale rekordy te należy interpretować jako częściowo dopasowane do materiału referencyjnego.

Przepływ przygotowania etykiet relewancji wyglądał następująco:

```text
pytanie + section_id + ground_truth_text
                    ↓
  ekstrakcja evidence_texts przy pomocy LLM
                    ↓
     ręczna i automatyczna weryfikacja
                    ↓
  mapowanie evidence na chunki z dane.csv
                    ↓
              relevant chunks
```

### Mapowanie evidence na relevant chunks

Zweryfikowane evidence odnosi się do tekstu źródłowego, natomiast system wyszukuje w 735 gotowych chunkach z `dane.csv`. Dlatego dla każdego pytania najpierw ograniczałem kandydatów do chunków pochodzących z dokumentu wskazanego przez `qrels`, a następnie porównywałem ich treść z każdym fragmentem evidence.

Dla każdego chunka obliczane jest `evidence_coverage`, czyli procent całego evidence dla pytania, który można odnaleźć w tym chunku. Chunk zostaje zapisany jako relevant, jeżeli osiąga ustalony minimalny próg. Ponieważ evidence może być rozłożone na kilka chunków, golden dataset przechowuje listę wszystkich chunków spełniających kryterium.

Dodatkowo obliczane jest `evidence_coverage_percentage` dla całego pytania. Wartość ta pokazuje, jaki procent evidence pokrywa suma wybranych relevant chunks. Pokrycie jest liczone jako unia dopasowanych zakresów, dlatego ten sam tekst znaleziony w kilku chunkach nie jest zaliczany wielokrotnie.

Oddzielenie evidence od granic chunków było celową decyzją: przy zmianie strategii chunkowania adnotacje źródłowe pozostają aktualne i mogą zostać ponownie zmapowane. Relevant chunks są więc wynikiem mapowania evidence na konkretny korpus, a nie pierwotnym źródłem ground truth.

### Struktura golden datasetu

Finalny `golden_dataset.json` jest tablicą 50 rekordów. Każdy rekord łączy dane pytania z Open RAGBench, evidence wyekstrahowane przy pomocy LLM i zweryfikowane względem źródła oraz listę chunków uznanych za relewantne:

```json
{
  "query_id": "...",
  "question": "...",
  "type": "abstractive",
  "source": "text",
  "doc_id": "...",
  "filename": "....pdf",
  "section_id": 17,
  "ground_truth_text": "...",
  "evidence_text": "...",
  "evidence_coverage_percentage": 100.0,
  "relevant_chunks": [
    {
      "chunk_id": "....pdf_0013",
      "evidence_coverage": 1.0
    }
  ]
}
```

Znaczenie poszczególnych pól jest następujące:

| Pole | Znaczenie |
| --- | --- |
| `query_id` | Unikalny identyfikator pytania zachowany z Open RAGBench. |
| `question` | Treść pytania przekazywana później do retrievera. |
| `type` | Typ pytania określony w benchmarku, np. `abstractive`. |
| `source` | Modalność źródła pytania; w wybranym podzbiorze pozostawiono pytania tekstowe. |
| `doc_id` | Identyfikator właściwego dokumentu w Open RAGBench. |
| `filename` | Nazwa PDF-u odpowiadającego dokumentowi i chunkom w `dane.csv`. |
| `section_id` | Identyfikator sekcji wskazanej przez relację `qrels`. |
| `ground_truth_text` | Pełny tekst referencyjnej sekcji Open RAGBench. Pole pozwala zachować pochodzenie evidence i ponownie zweryfikować adnotację. |
| `evidence_text` | Fragmenty potrzebne do odpowiedzi, wyekstrahowane przy pomocy LLM i zweryfikowane względem źródła. Jeżeli w `evidence_annotations.json` było ich kilka, w finalnym rekordzie są łączone spacją. |
| `evidence_coverage_percentage` | Procent evidence pokryty łącznie przez wszystkie wybrane relevant chunks. |
| `relevant_chunks` | Lista chunków spełniających minimalny próg coverage, uporządkowana malejąco według dopasowania. |
| `relevant_chunks[].chunk_id` | Stabilny identyfikator chunka w postaci nazwy pliku i kolejnego numeru. |
| `relevant_chunks[].evidence_coverage` | Część całego evidence pokryta przez pojedynczy chunk, zapisana w zakresie od `0` do `1`. |

Taka struktura zawiera zarówno dane potrzebne do uruchomienia ewaluacji (`query_id`, `question`, `relevant_chunks`), jak i informacje umożliwiające audyt sposobu utworzenia ground truth (`ground_truth_text`, `evidence_text` i wartości coverage). Metryki retrievalowe będą porównywać identyfikatory chunków zwróconych przez system z `chunk_id` zapisanymi w `relevant_chunks`.

### Problemy z dopasowaniem tekstu

Tekst benchmarku zawiera Markdown i LaTeX, natomiast `dane.csv` powstał przez ekstrakcję tekstu z PDF. Powodowało to różnice w symbolach, odstępach i kolejności indeksów matematycznych, na przykład `Q_i^(nom)` oraz `Q (nom) i`. Proste porównanie znaków zaniżało coverage mimo obecności właściwego fragmentu.

Rozszerzyłem więc normalizację tekstu o obsługę LaTeX-u, symboli matematycznych, wariantów minusa, liczb rozdzielonych spacjami i słów przerwanych końcem linii. Dla lokalnych wyrażeń matematycznych dodałem dopasowanie tokenów odporne na zmianę kolejności indeksów. W zwykłej prozie kolejność słów nadal jest respektowana, aby ograniczyć fałszywe trafienia.

### Dobór progu coverage

Próg większy od zera okazał się zbyt liberalny, ponieważ krótkie wspólne frazy dodawały słabo powiązane chunki. Przetestowałem wartości od `0.00` do `0.90`, analizując zarówno liczbę wybranych chunków, jak i łączne pokrycie evidence dla całego pytania.

| Próg pojedynczego chunka | Chunki | Śr. chunków/pytanie | Bez chunków | Śr. coverage pytania | Minimum | Pytania <90% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,00 | 2615 | 52,30 | 0 | 99,76% | 90,98% | 0 |
| 0,05 | 2169 | 43,38 | 0 | 99,76% | 90,98% | 0 |
| 0,10 | 1450 | 29,00 | 0 | 99,76% | 90,98% | 0 |
| 0,15 | 966 | 19,32 | 0 | 99,76% | 90,98% | 0 |
| 0,20 | 578 | 11,56 | 0 | 99,76% | 90,98% | 0 |
| **0,25** | **364** | **7,28** | **0** | **99,76%** | **90,98%** | **0** |
| 0,30 | 226 | 4,52 | 0 | 99,06% | 79,37% | 2 |
| 0,35 | 151 | 3,02 | 0 | 98,35% | 73,03% | 3 |
| 0,40 | 128 | 2,56 | 0 | 97,71% | 67,87% | 4 |
| 0,50 | 91 | 1,82 | 1 | 93,40% | 0% | 10 |
| 0,60 | 66 | 1,32 | 1 | 92,28% | 0% | 11 |
| 0,75 | 45 | 0,90 | 9 | 79,95% | 0% | 14 |
| 0,90 | 37 | 0,74 | 15 | 69,78% | 0% | 15 |

Wybrałem `MIN_CHUNK_COVERAGE = 0.25`. Dla tego progu uzyskano:

- 364 relevant chunks,
- średnio 7,28 chunka na pytanie,
- średnie łączne coverage 99,76%,
- minimalne łączne coverage 90,98%,
- brak pytań bez relevant chunks.

Wartość `0.25` usunęła około 86% początkowych kandydatów bez obniżenia coverage względem niższych progów. Próg `0.30` ograniczał liczbę dopasowań jeszcze bardziej, ale obniżał minimalne coverage pytania do 79,37%, ponieważ usuwał również chunki zawierające unikalne części evidence. Wartość `0.25` została więc wybrana jako kompromis między ograniczeniem szumu a zachowaniem kompletnej informacji. Tabela jest również przechowywana osobno w `relevant_chunk_coverage_thresholds.md`.

### Walidacja końcowa

Pipeline sprawdza zgodność i unikalność `query_id`, obecność evidence, jego dokładne występowanie w `ground_truth_text`, brak duplikatów, zgodność dokumentów oraz spełnienie progu przez relevant chunks. Finalny `golden_dataset.json` zawiera 50 ręcznie zweryfikowanych rekordów i żaden z nich nie pozostaje bez relevant chunka.


## Stan realizacji i dalsze etapy

Zrealizowane zostały zadania obejmujące research, porównanie narzędzi, wybór danych, uruchomienie środowiska, indeksowanie, retrieval oraz przygotowanie golden datasetu. Projekt ma odtwarzalną podstawę do rozpoczęcia właściwych eksperymentów.

Do wykonania pozostają:

- implementacja Precision@k, Recall@k, MRR i nDCG@k,
- ewaluacja Faithfulness i Answer Relevance,
- porównanie różnych wartości `top_k`,
- dodanie i ocena rerankera `BAAI/bge-reranker-v2-m3`,
- automatyczne generowanie tabeli wyników dla wielu konfiguracji,
- przygotowanie dashboardu lub raportu końcowego.

Najważniejszym rezultatem obecnego etapu jest spójne środowisko oraz zweryfikowany zbiór referencyjny. Bez dobrego golden datasetu dalsze metryki byłyby łatwe do policzenia, ale ich wyniki nie byłyby wiarygodne.
