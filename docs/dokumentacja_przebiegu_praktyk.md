# Dokumentacja przebiegu praktyk zawodowych

## Cel praktyk

Celem pracy było przygotowanie podstaw prototypu służącego do porównywania jakości retrievalu w systemach RAG. Zakres został podzielony w Jirze na cztery obszary: research, przygotowanie środowiska, budowę pipeline'u ewaluacyjnego oraz podsumowanie wyników.

W ramach dotychczasowych prac:

- przeanalizowałem metody ewaluacji systemów RAG,
- porównałem dostępne narzędzia ewaluacyjne,
- wybrałem publiczny korpus testowy,
- przygotowałem lokalne środowisko z PostgreSQL i pgvector,
- uruchomiłem indeksowanie i wyszukiwanie wektorowe,
- zbudowałem ręcznie zweryfikowany golden dataset,
- zaimplementowałem i przetestowałem metryki retrievalu oraz agregację wyników.

Kolejne etapy obejmują uruchamianie pełnych eksperymentów, ewaluację generowania odpowiedzi i raportowanie wyników.

## Analiza metod i metryk ewaluacji RAG

Pierwszym etapem było zapoznanie się z metodami oceny systemów RAG oraz wybór metryk odpowiednich dla planowanego prototypu. W ramach analizy wyróżniłem dwa obszary, które w przyszłości powinny być oceniane osobno:

1. **Retrieval** — sprawdzenie, czy system znajduje właściwe fragmenty i umieszcza je odpowiednio wysoko w rankingu.
2. **Generation** — sprawdzenie, czy odpowiedź jest zgodna z pobranym kontekstem i odpowiada na pytanie.

Dla ewaluacji retrievalu zaimplementowałem **Precision@k, Recall@k, HitRate@k, MRR, nDCG@k, graded nDCG@k, weighted Precision@k i EvidenceCoverage@k**. Weighted Precision@k jest normalizowane względem sumy ocen idealnego top-k, dzięki czemu idealny zestaw wyników otrzymuje `1.0` niezależnie od rozłożenia evidence między chunkami. Dodałem również `RetrievalEvaluator`, agregację wyników wielu pytań i testy jednostkowe. Do późniejszej oceny generowania wybrałem **Faithfulness** i **Answer Relevance**.

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

Embedding service udostępnia endpoint zwracający nazwę modelu i rozmiar wektora. Backend pobiera te dane, dzięki czemu wymiar embeddingu nie jest hardkodowany. Dla aktualnego modelu `BAAI/bge-m3` baza używa typu `vector(1024)`.

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

Adnotacje zostały następnie kilkukrotnie zweryfikowane pod względem kompletności, przydatności, pełności zdań i dokładnej zgodności z tekstem źródłowym. Każdy niepusty fragment został też automatycznie sprawdzony jako substring właściwego `ground_truth_text`. Ostatecznie wszystkie 50 rekordów otrzymało evidence. Mediana evidence wynosi 240,5 znaku, czyli około 23% długości medianowego chunka i około 13% medianowego `ground_truth_text`.

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

Dla każdego chunka obliczane jest `evidence_coverage`, czyli część całego evidence dla pytania obecna w tym chunku. Relevance jest ustalana osobno dla każdego fragmentu `evidence_texts`, a nie przez jeden globalny próg coverage. Uwzględniane są lokalne dopasowania tekstu, evidence przecięte granicą chunków oraz osobne reguły dla wzorów matematycznych.

Dodatkowo builder sprawdza łączne pokrycie evidence przez wszystkie wybrane relevant chunks. Pokrycie jest liczone jako unia dopasowanych zakresów, dlatego ten sam tekst znaleziony w kilku chunkach nie jest zaliczany wielokrotnie. Każde pytanie musi osiągnąć pełne pokrycie; wynik tej walidacji nie jest zapisywany w finalnym pliku.

Oddzielenie evidence od granic chunków było celową decyzją: przy zmianie strategii chunkowania adnotacje źródłowe pozostają aktualne i mogą zostać ponownie zmapowane. Relevant chunks są więc wynikiem mapowania evidence na konkretny korpus, a nie pierwotnym źródłem ground truth.

### Struktura golden datasetu

Finalny `golden_dataset.json` zawiera wspólne metadane i 50 rekordów potrzebnych do ewaluacji. Dane audytowe pozostają w plikach procesu przygotowania i nie są powielane w kontrakcie backendu:

```json
{
  "metadata": {
    "schema_version": 1,
    "evidence_interval_gap_tolerance": 3
  },
  "records": [
    {
      "query_id": "...",
      "question": "...",
      "expected_answer": "...",
      "evidence": [
        {
          "text": "...",
          "normalized_length": 120
        }
      ],
      "relevant_chunks": [
        {
          "chunk_id": "....pdf_0013",
          "evidence_coverage": 1.0,
          "evidence_intervals": [
            {
              "evidence_index": 0,
              "intervals": [[0, 120]]
            }
          ]
        }
      ]
    }
  ]
}
```

Znaczenie poszczególnych pól jest następujące:

| Pole | Znaczenie |
| --- | --- |
| `metadata.schema_version` | Wersja kontraktu pliku. |
| `metadata.evidence_interval_gap_tolerance` | Wspólna tolerancja scalania przedziałów dla wszystkich rekordów. |
| `records[].query_id` | Unikalny identyfikator pytania zachowany z Open RAGBench. |
| `records[].question` | Treść pytania przekazywana do retrievera. |
| `records[].expected_answer` | Odpowiedź referencyjna używana później w ewaluacji generowania. |
| `records[].evidence` | Lista oddzielnych, zweryfikowanych fragmentów evidence wraz z ich znormalizowaną długością. |
| `records[].evidence[].text` | Oryginalna treść zweryfikowanego fragmentu evidence. |
| `records[].evidence[].normalized_length` | Długość znormalizowanego fragmentu, używana przy obliczaniu `EvidenceCoverage@k`. |
| `records[].relevant_chunks` | Lista chunków zawierających znaczącą część co najmniej jednego fragmentu evidence. |
| `records[].relevant_chunks[].chunk_id` | Stabilny identyfikator chunka w postaci nazwy pliku i kolejnego numeru. |
| `records[].relevant_chunks[].evidence_coverage` | Część całego evidence pokryta przez pojedynczy chunk, zapisana w zakresie od `0` do `1`. |
| `records[].relevant_chunks[].evidence_intervals` | Przedziały znormalizowanego evidence pokrywane przez chunk, używane do obliczania unii pokrycia dla top-k. |

Taka struktura zawiera zarówno dane potrzebne do uruchomienia ewaluacji, jak i informacje umożliwiające audyt sposobu utworzenia ground truth. Metryki binarne porównują identyfikatory zwróconych chunków z `relevant_chunks`, metryki stopniowane wykorzystują `evidence_coverage`, a `EvidenceCoverage@k` scala zapisane przedziały i nie liczy tego samego fragmentu wielokrotnie.

### Problemy z dopasowaniem tekstu

Tekst benchmarku zawiera Markdown i LaTeX, natomiast `dane.csv` powstał przez ekstrakcję tekstu z PDF. Powodowało to różnice w symbolach, odstępach i kolejności indeksów matematycznych, na przykład `Q_i^(nom)` oraz `Q (nom) i`. Proste porównanie znaków zaniżało coverage mimo obecności właściwego fragmentu.

Rozszerzyłem normalizację o obsługę LaTeX-u, symboli matematycznych, wariantów minusa, liczb rozdzielonych spacjami i słów przerwanych końcem linii. Matcher łączy tylko lokalne, uporządkowane bloki tekstu. Duże fragmenty LaTeX są porównywane osobno z kontrolą wartości liczbowych i tokenów kotwiczących, co ogranicza dopasowania podobnych, ale różnych wzorów.

### Reguły relevance i coverage

Początkowo relevance wyznaczał globalny próg `MIN_CHUNK_COVERAGE = 0.25`. Rozwiązanie powodowało false positives dla często powtarzających się fraz i podobnych wzorów matematycznych, dlatego zostało zastąpione oceną każdego fragmentu evidence osobno.

Aktualne reguły wymagają co najmniej 30 dopasowanych znaków oraz odpowiednio: 80% lokalnego dopasowania, 40% jednego spójnego bloku albo 35% przy podziale evidence na granicy chunków. Wzory matematyczne wymagają 85% zgodności tokenów oraz zgodności liczb i tokenów kotwiczących.

Aktualny golden dataset zawiera 50 pytań i 77 relevant chunks. Średnia wynosi 1,54 chunka na pytanie, mediana 1, maksimum 4. Każde pytanie ma co najmniej jeden relevant chunk i 100% łącznego evidence coverage.

### Walidacja końcowa

Builder sprawdza zgodność i unikalność `query_id`, obecność odpowiedzi i evidence, dokładne występowanie evidence w `ground_truth_text`, zgodność dokumentów oraz brak pytań bez relevant chunks. Finalny `golden_dataset.json` zawiera 50 zweryfikowanych rekordów i 100% łącznego evidence coverage dla każdego pytania.


## Stan realizacji i dalsze etapy

Zrealizowane zostały zadania obejmujące research, wybór danych, środowisko, indeksowanie, retrieval, golden dataset, metryki retrievalu, evaluator i agregację wyników.

Do wykonania pozostają:

- wczytywanie golden datasetu w backendzie i połączenie go z retrieval oraz evaluatorem,
- implementacja generowania odpowiedzi i ewaluacja Faithfulness oraz Answer Relevance,
- porównanie różnych wartości `top_k`,
- dodanie i ocena rerankera `BAAI/bge-reranker-v2-m3`,
- automatyczne generowanie tabeli wyników dla wielu konfiguracji,
- przygotowanie dashboardu lub raportu końcowego.

Najważniejszym rezultatem obecnego etapu jest spójne środowisko oraz zweryfikowany zbiór referencyjny. Bez dobrego golden datasetu dalsze metryki byłyby łatwe do policzenia, ale ich wyniki nie byłyby wiarygodne.
