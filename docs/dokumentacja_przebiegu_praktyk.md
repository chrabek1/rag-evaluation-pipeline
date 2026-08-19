# Dokumentacja przebiegu praktyk zawodowych

Celem praktyk jest przygotowanie prototypu służącego do porównywania jakości różnych konfiguracji systemu RAG. Zakres prac odpowiada czterem epikom zdefiniowanym w Jirze: Research, Setup prototypu, Pipeline ewaluacyjny i Podsumowanie.

## 1. Research

### 1.1. Sposób ewaluacji i wybór narzędzi

Ewaluację systemu RAG można podzielić na dwa osobne obszary:

1. **Retrieval** — ocena, czy system znajduje właściwe fragmenty dokumentów i umieszcza je odpowiednio wysoko w rankingu.
2. **Generation** — ocena, czy wygenerowana odpowiedź odpowiada na pytanie i jest zgodna z dostarczonym kontekstem.

Do klasycznej oceny retrievalu można wykorzystać:

- `Precision@k` — część wyników top-k, które są relewantne;
- `Recall@k` — część wszystkich relewantnych chunków odnaleziona w top-k;
- `HitRate@k` — informację, czy w top-k znajduje się przynajmniej jeden relewantny chunk;
- `MRR@k` — ocenę pozycji pierwszego relewantnego wyniku;
- `nDCG@k` — ocenę kolejności wszystkich relewantnych wyników.

Metryki binarne nie rozróżniają chunka zawierającego całą potrzebną informację od chunka pokrywającego tylko jej część. Dlatego w ewaluacji można wykorzystać również stopień pokrycia evidence przez pojedynczy chunk:

- `graded nDCG@k` pozwala uwzględnić zarówno coverage, jak i pozycję chunka;
- znormalizowane `weighted Precision@k` pozwala porównać jakość znalezionego zestawu z najlepszym możliwym zestawem top-k;
- `EvidenceCoverage@k` pozwala obliczyć unię odnalezionych przedziałów evidence, bez wielokrotnego zaliczania tego samego fragmentu obecnego w kilku chunkach.

W ramach researchu porównałem RAGAS, DeepEval, TruLens i Arize Phoenix. Narzędzia te są szczególnie użyteczne w ocenie generation, ponieważ wspierają metryki semantyczne i podejście LLM-as-a-Judge. RAGAS wybrałem jako główne narzędzie do dalszej pracy ze względu na koncentrację na ewaluacji RAG oraz dostępność metryk takich jak Faithfulness i Answer Relevance.

RAGAS może być także użyteczny w ocenie retrievalu, między innymi przez metryki Context Precision i Context Recall. Są to jednak oceny semantyczne, często zależne od modelu oceniającego. RAGAS nie zastępuje prostych, deterministycznych metryk opartych na identyfikatorach chunków, ich pozycji i własnych adnotacjach coverage. Takie metryki łatwiej kontrolować, interpretować i rozszerzać o zasady specyficzne dla przygotowanego golden datasetu.

Z tego powodu przyjąłem następujący podział:

- ewaluację retrievalu zaimplementuję samodzielnie z użyciem klasycznych i własnych metryk opartych na evidence coverage;
- do ewaluacji generation wykorzystam RAGAS;
- oba etapy połączę w jeden pipeline end-to-end, zachowując osobne wyniki.

Pozwoli to ustalić nie tylko, która konfiguracja generuje lepsze odpowiedzi, lecz również czy jej wynik wynika z jakości retrievalu, czy z możliwości modelu generującego. Szczegółowy research znajduje się w `rag_evaluation_research.md`, a porównanie narzędzi w `rag_evaluation_tools_comparison.md`.

### 1.2. Wybór danych testowych

Jako publiczne dane testowe wybrałem Open RAGBench (`deepmatics/open_ragbench`) udostępniony na Hugging Face. Benchmark zawiera dane przygotowane na podstawie dokumentów z kilku domen. W projekcie wykorzystałem część opartą na publikacjach naukowych z arXiv.

Open RAGBench udostępnia:

- dokumenty źródłowe i adresy plików PDF;
- ustrukturyzowany tekst dokumentów podzielony na sekcje;
- pytania przypisane do dokumentów;
- odpowiedzi referencyjne;
- relacje `qrels`, które wskazują dokument i sekcję powiązaną z pytaniem;
- informację o typie źródła, między innymi tekstowym lub wymagającym interpretacji obrazu.

Taka struktura pozwala wykorzystać benchmark jako podstawę golden datasetu: pytanie może zostać wysłane do retrievera, odpowiedź referencyjna użyta w ewaluacji generation, a wskazana sekcja dokumentu stanowi źródło do wyznaczenia właściwych fragmentów. Samo `qrels` nie daje jednak gotowych identyfikatorów chunków dla własnego sposobu chunkowania, dlatego sekcje źródłowe wymagają dalszego doprecyzowania i zmapowania na korpus używany przez prototyp.

## 2. Setup prototypu

### 2.1. Przygotowanie podzbioru dokumentów

Za pomocą skryptu `prepare_open_rag_subset.py` wybrałem podzbiór Open RAGBench o łącznej objętości możliwie bliskiej 700 000 znaków. Limit znaków pozwolił lepiej kontrolować wielkość danych niż sama liczba dokumentów, ponieważ publikacje znacznie różnią się długością.

Wraz z dokumentami wybrałem wszystkie przypisane do nich pytania. Pominąłem pytania typu `text-image` i `text-table-image`, ponieważ otrzymane chunki zawierają wyłącznie tekst i nie zachowują informacji z obrazów ani tabel zapisanych jako obrazy. Pozostawienie takich pytań prowadziłoby do przypadków, których retriever tekstowy nie mógłby poprawnie rozwiązać.

Finalny podzbiór obejmuje:

- 9 dokumentów PDF;
- 698 275 znaków tekstu;
- 50 pytań tekstowych.

Plik `dane.csv` zawierał 735 gotowych chunków. W prototypie nie wykonywałem ponownego chunkowania, lecz indeksowałem dostarczone fragmenty.

Wybór jest zapisany w `selected_documents.json`, a pytania wraz z odpowiedziami i sekcjami źródłowymi w `selected_questions.json`. Dzięki temu ten etap można odtworzyć bez ponownego ręcznego wybierania danych.

### 2.2. Architektura i sposób tworzenia prototypu

Prototyp działa w środowisku Docker Compose i jest podzielony na dwa serwisy aplikacyjne:

- `backend` odpowiada za wczytywanie danych, indeksowanie, komunikację z bazą, retrieval i ewaluację;
- `embedding_service` udostępnia osobną usługę generującą embeddingi przy użyciu modelu `BAAI/bge-m3`.

Dodatkowym elementem infrastruktury jest PostgreSQL z rozszerzeniem pgvector, który przechowuje embeddingi 735 chunków i wykonuje wyszukiwanie wektorowe. Embedding service udostępnia endpoint zwracający nazwę modelu i wymiar embeddingu, dlatego backend nie ma zahardkodowanego rozmiaru wektora. Dla aktualnego modelu baza korzysta z typu `vector(1024)`.

Podczas tworzenia prototypu korzystałem z ChatGPT i Codexa. Kod `backend` i `embedding_service` przepisywałem ręcznie, funkcja po funkcji. Pozwalało mi to dokładnie rozumieć działanie rozwiązania i świadomie podejmować decyzje architektoniczne. Testy oraz pomocnicze skrypty służące do przygotowania i walidacji datasetu były w większości generowane przez Codex i wymagały z mojej strony znacznie mniejszej ingerencji.

### 2.3. Przygotowanie golden datasetu

Open RAGBench dostarcza pytanie, odpowiedź referencyjną, identyfikator dokumentu, `section_id` oraz tekst odpowiadającej mu sekcji. W projekcie tekst sekcji jest zachowywany jako `ground_truth_text`. Sekcja wskazuje obszar dokumentu, na podstawie którego utworzono pytanie, ale nie określa jeszcze, które chunki z `dane.csv` powinny zostać uznane za relewantne.

#### 2.3.1. Evidence

Granice sekcji Open RAGBench nie pokrywają się z granicami 735 gotowych chunków. Sekcje są też istotnie większe od chunków. Jedna sekcja może obejmować wiele chunków, również takich, które nie zawierają informacji potrzebnych do odpowiedzi. Uznanie ich wszystkich za relewantne prowadziłoby do zbyt szerokich etykiet i zawyżania jakości retrievalu.

Dlatego zdecydowałem się stworzyć evidence, czyli krótkie fragmenty `ground_truth_text` zawierające informacje potrzebne do udzielenia odpowiedzi. Przygotowałem je z pomocą LLM, następnie zweryfikowałem i zapisałem w `evidence_annotations.json`.

Porównanie długości tekstów pokazuje różnicę między pełną sekcją, chunkami i evidence:

| Rodzaj tekstu | Liczba | Minimum | Mediana | Średnia | Maksimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chunki | 735 | 105 | 1 036 | 976,20 | 2 022 |
| `ground_truth_text` | 50 | 328 | 1 828,50 | 2 824,46 | 13 575 |
| Evidence łącznie dla pytania | 50 | 76 | 240,50 | 361,40 | 1 199 |

Średni `ground_truth_text` jest około 2,9 razy dłuższy od przeciętnego chunka, a jego mediana jest 1,76 raza większa od mediany chunka.

Najdłuższy `ground_truth_text` ma 13 575 znaków i obejmuje 12 chunków. Evidence ma 147 znaków i mieści się w chunku `0049`.

```text
ground_truth_text
├───────────────────────────────────────────────────────────┤
0                                                      13 575

chunki
├0038┼0039┼0040┼0041┼0042┼0043┼0044┼0045┼0046┼0047┼0048┼0049┤

evidence
                                                       ├E1┤
                                       E1: 147 znaków w chunku 0049
```

Schemat przedstawia kolejność fragmentów, a nie dokładną skalę ich długości. Evidence zajmuje około 1,1% całego tekstu, dlatego zostało powiększone.

Podczas tworzenia evidence przyjąłem następujące zasady:

- evidence może pochodzić wyłącznie z `ground_truth_text` przypisanego do danego pytania;
- każdy fragment musi być dokładnym substringiem tekstu źródłowego;
- nie wolno parafrazować, poprawiać treści ani dodawać wiedzy zewnętrznej;
- evidence powinno zawierać wszystkie informacje potrzebne do odpowiedzi, w tym warunki, ograniczenia i wyjątki;
- informacje pochodzące z kilku miejsc należy zapisać jako osobne fragmenty;
- należy zachowywać pełne zdania i nie usuwać kontekstu potrzebnego do ich interpretacji;
- każdy fragment powinien zostać sprawdzony w kontekście pytania, odpowiedzi referencyjnej i dokumentu PDF.

#### 2.3.2. Chunk relevance

Chunk jest uznawany za relewantny na podstawie pokrycia evidence, a nie samej przynależności do sekcji.

Dla pojedynczego chunka coverage jest liczone jako:

```text
evidence_coverage =
    liczba pokrytych znaków znormalizowanego evidence
    / łączna liczba znaków wszystkich znormalizowanych fragmentów evidence
```

| Właściwość | Wartość |
| --- | ---: |
| Liczba pytań | 50 |
| Liczba fragmentów evidence | 96 |
| Liczba relewantnych chunków | 77 |
| Średnia liczba relewantnych chunków na pytanie | 1,54 |
| Mediana relewantnych chunków na pytanie | 1 |
| Zakres liczby relewantnych chunków | 1–4 |
| Średnie coverage pojedynczego chunka | 71,1% |
| Mediana coverage pojedynczego chunka | 77,7% |
| Łączne coverage evidence dla każdego pytania | 100% |

Pełne zestawienie dla poszczególnych pytań znajduje się w pliku [`golden_dataset_relevance.md`](golden_dataset_relevance.md).

Zapisane przedziały pozwalają liczyć unię pokrycia i nie zaliczać wielokrotnie tej samej części evidence dopasowanej do kilku chunków.

#### 2.3.3. Struktura golden datasetu

Finalny `golden_dataset.json` zawiera wspólne metadane i listę rekordów:

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
          "chunk_id": "document.pdf_0013",
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

`question` jest wejściem retrievera, `expected_answer` może zostać użyte w ewaluacji generation, `evidence` zachowuje zweryfikowane fragmenty źródłowe, a `relevant_chunks` dostarcza etykiety i wartości potrzebne do metryk retrievalu. Finalny dataset obejmuje 50 pytań i 77 relevant chunks. Każde pytanie ma co najmniej jeden relevant chunk oraz pełne łączne pokrycie evidence.

## 3. Pipeline ewaluacyjny

### 3.1. Ewaluacja retrievalu

Dla każdego pytania retriever zwraca ranking top-k chunków. Porównuję go z `relevant_chunks` zapisanymi w golden datasecie, obliczam metryki dla pojedynczego pytania, a następnie agreguję wyniki dla całego zbioru.

Do podstawowej oceny wykorzystuję metryki binarne, w których chunk jest relewantny albo nierelewantny:

| Metryka | Znaczenie |
| --- | --- |
| `Precision@k` | Jaka część zwróconych top-k chunków jest relewantna. |
| `Recall@k` | Jaka część wszystkich relewantnych chunków została odnaleziona w top-k. |
| `HitRate@k` | Czy w top-k znajduje się przynajmniej jeden relewantny chunk. |
| `MRR@k` | Jak wysoko znajduje się pierwszy relewantny wynik. Dla pojedynczego pytania liczony jest Reciprocal Rank, a MRR jest jego średnią dla wszystkich pytań. |
| `nDCG@k` | Czy wszystkie relewantne chunki znajdują się możliwie wysoko w rankingu. Niższe pozycje mają mniejszą wagę. |

Przypisanie każdemu relewantnemu chunkowi wartości `evidence_coverage` pozwala ocenić nie tylko, czy chunk jest trafny, ale również jaką część informacji potrzebnej do odpowiedzi zawiera. Dzięki temu można uwzględnić stopień relewantności, kolejność wyników i łączne pokrycie evidence w metrykach `Graded nDCG@k`, `Weighted Precision@k` oraz `EvidenceCoverage@k`.

| Metryka | Znaczenie |
| --- | --- |
| `Graded nDCG@k` | Ocenia kolejność wyników, używając coverage jako stopnia relewantności. Najwyżej powinny znajdować się chunki pokrywające największą część evidence. |
| `Weighted Precision@k` | Porównuje sumę coverage zwróconych chunków z najlepszą możliwą sumą coverage dla top-k. Wynik jest znormalizowany, dlatego pełne odnalezienie najlepszego zestawu daje `1.0`. |
| `EvidenceCoverage@k` | Mierzy, jaka część całego evidence została pokryta przez unię chunków z top-k. Ten sam przedział evidence dopasowany do kilku chunków jest liczony tylko raz. |

Wszystkie metryki przyjmują wartości od `0.0` do `1.0`, a wyższy wynik oznacza lepszy retrieval. Klasyczne metryki oceniają trafność identyfikatorów i ranking, natomiast metryki stopniowane pokazują, ile informacji potrzebnej do odpowiedzi rzeczywiście znaleziono.
