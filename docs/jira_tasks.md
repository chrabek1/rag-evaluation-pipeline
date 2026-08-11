# Zadania z Jiry — treść do realizacji


## Epik 1 — Research

- ECW-1 — Research
  - Przegląd literatury dotyczącej ewaluacji chunkowania i retrievalu w RAG.

  - ECW-2 — Przegląd literatury nt. ewaluacji chunkowania i retrievalu w RAG

    Opis
    Zebranie kluczowych podejść i metryk do oceny jakości wyszukiwania w systemach RAG (precision@k, recall@k, MRR, NDCG, faithfulness, answer relevance). Przegląd typowych metodologii budowania golden datasetów oraz najczęstszych pułapek przy ewaluacji.

  - ECW-3 — Przegląd narzędzi do ewaluacji RAG (RAGAS, TruLens, DeepEval, Arize Phoenix)

    Opis
    Porównanie dostępnych narzędzi pod kątem: łatwości integracji z Chroma, wspieranych metryk, kosztu (czy wymagają dodatkowych wywołań LLM do oceny), aktywności projektu i jakości dokumentacji.

    Najlepiej jakby powstała jakaś tabela porównawcza narzędzi w ramach tego zadania. I z tych narzędzi wybrać 1-2 do dalszej pracy.

  - ECW-4 — Ustalenie zakresu eksperymentu i danych testowych

    Opis
    Wybór publicznego/otwartego zbioru dokumentów (np. dokumentacja techniczna, artykuły pomocy, Wikipedia) jako namiastki firmowej bazy wiedzy. Zbiór powinien być strukturalnie zbliżony (długie dokumenty, nagłówki, listy), ale w pełni neutralny i publicznie dostępny.

## Epik 2 — Setup prototypu

- ECW-5 — Setup prototypu

  - ECW-6 — Wybór dokumentów testowych i przekazanie do chunkowania

    Opis
    Wybór docelowego zbioru dokumentów (zgodnie z ustaleniami z zadania "Ustalenie zakresu eksperymentu i danych testowych"). Zgłoszenie wybranego zbioru do mnie, przygotuje podział dokumentów na chunki.

  - ECW-7 — Postawienie środowiska: lokalny pgvector+ skrypt indeksujący

    Opis
    Uruchomienie lokalnej instancji pgvector (Docker) oraz napisanie skryptu do zapisu gotowych chunków (otrzymanych od Arkadiusz Gajdy) jako embeddingów. Zarządzanie zależnościami przez uv.

    Kryteria akceptacji:
    - Jedna komenda indeksuje gotowe chunki do lokalnej instancji pgvector
    - Środowisko odtwarzalne (Dockerfile / docker-compose)

  - ECW-8 — Przygotowanie zbioru testowego (golden dataset)

    Opis
    Wygenerowanie lub zebranie 30-50 par pytanie-odpowiedź z adnotacją poprawnego fragmentu źródłowego. Można wspomóc się LLM-em do generacji pytań na podstawie dokumentów, z ręczną weryfikacją. To kluczowy element całego pipeline'u, od jakości tego zbioru zależy wiarygodność metryk.

    Kryteria akceptacji:
    - Gotowy zbiór testowy w formacie strukturalnym (np. JSON/CSV)
    - Każde pytanie ma wskazany poprawny fragment źródłowy
    - Zbiór zweryfikowany ręcznie pod kątem jakości
    - Krótki opis metodologii tworzenia zbioru, do odtworzenia na danych firmowych

  - ECW-9 — Zaprojektowanie narzędzia jako reużywalnego (config-driven)

    Opis
    Upewnienie się, że cały pipeline (indeksacja, model embeddingowy, źródło danych, format golden datasetu) jest sterowany konfiguracją, a nie zahardkodowany pod konkretny zbiór testowy. Celem jest, żeby narzędzie dało się później podłączyć do innych danych i innego modelu embeddingowego bez przepisywania kodu.

    Kryteria akceptacji:
    - Wszystkie kluczowe parametry (ścieżka do chunków, model embeddingowy, connection string do pgvector, ścieżka do golden datasetu) w jednym pliku konfiguracyjnym / zmiennych środowiskowych
    - Krótki README opisujący, jak podmienić dane wejściowe na inne
    - Brak twardo zapisanych wartości w kodzie logiki

## Epik 3 — Pipeline ewaluacyjny

- ECW-10 — Pipeline ewaluacyjny

  - ECW-11 — Implementacja metryk retrievalu

    Opis
    Implementacja lub integracja (przez narzędzie z research) metryk oceniających jakość wyszukiwania: precision@k, recall@k, MRR, NDCG. To główny komponent całego prototypu.

    Kryteria akceptacji:
    - Pipeline automatycznie liczy metryki dla danego zbioru testowego
    - Wyniki zapisywane w ustrukturyzowanej formie (np. CSV/JSON)
    - Kod zorganizowany tak, by dało się łatwo podmienić model embeddingowy / parametry bez zmian w logice liczenia metryk

  - ECW-12 — Metryki end-to-end jakości generacji

    Opis
    Ocena, czy odpowiedź modelu językowego jest wsparta znalezionym kontekstem (faithfulness, answer relevance). Można użyć lokalnego modelu open-source lub modelu chmurowego który udostępnia darmowe użycie np Gemini.

    Kryteria akceptacji:
    - Pipeline zwraca dodatkowe metryki generacji obok metryk retrievalu

  - ECW-13 — Automatyzacja porównania konfiguracji

    Opis
    Skrypt uruchamiający pełny cykl ewaluacji (indeksacja do pgvector, retrieval, liczenie metryk) dla kilku wariantów na raz, a następnie przeprowadzenie na nim eksperymentów. Model embeddingowy: BAAI/bge-m3. Jako reranker: model BAAI/bge-reranker-v2-m3. Testowane warianty: różne wartości top-k, z rerankingiem i bez.

    Kryteria akceptacji:
    - Jedno uruchomienie generuje tabelę/wykres porównawczy dla wszystkich testowanych wariantów
    - Model embeddingowy (BAAI/bge-m3) i reranker (BAAI/bge-reranker-v2-m3) jasno wskazane w README
    - Łatwe dodanie nowego wariantu top-k/rerankingu do porównania bez przepisywania pipeline'u
    - Zebrane wyniki dla min. 4-5 konfiguracji (różne top-k, z/bez rerankingu)

## Epik 4 — Podsumowanie

- ECW-14 — Podsumowanie

  - ECW-15 — Dashboard/raport wizualizujący wyniki

    Opis
    Prosty dashboard (np. Streamlit) lub notebook prezentujący wykresy porównujące konfiguracje pod kątem zebranych metryk.

    Kryteria akceptacji:
    - Czytelna wizualizacja pokazująca, która konfiguracja wypada najlepiej i dlaczego

  - ECW-16 — Dokumentacja końcowa

    Opis
    Podsumowanie: jaka metodologia oceny sprawdziła się najlepiej, jakie metryki warto na stałe wdrożyć jako narzędzie do regularnego sprawdzania jakości retrievalu, jakie wnioski płyną z przeprowadzonych eksperymentów.

    Kryteria akceptacji:
    - Dokument podsumowujący z wnioskami i rekomendacjami
    - Kod prototypu
