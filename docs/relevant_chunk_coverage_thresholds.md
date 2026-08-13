# Wpływ progu coverage chunka na coverage pytania

Tabela przedstawia wpływ minimalnego `evidence_coverage` pojedynczego chunka na liczbę chunków zachowanych w golden datasecie oraz łączne pokrycie evidence dla 50 pytań.

| Próg pojedynczego chunka | Chunki | Śr. chunków/pytanie | Bez chunków | Śr. coverage pytania | Minimum | Pytania <90% |
|---:|---:|---:|---:|---:|---:|---:|
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

## Rekomendacja

Rekomendowany próg wynosi **0,25**. Usuwa około 86% początkowych kandydatów, zachowując średnie coverage pytania na poziomie 99,76% i minimalne coverage na poziomie 90,98%. Żadne pytanie nie traci wszystkich relevant chunks ani nie spada poniżej 90% łącznego pokrycia evidence.

Podniesienie progu do 0,30 redukuje liczbę chunków do 226, ale zaczyna usuwać chunki zawierające unikalne części evidence: minimalne coverage pytania spada wtedy do 79,37%.

Wyniki zostały obliczone z użyciem aktualnej normalizacji tekstu oraz lokalnego, niewrażliwego na kolejność dopasowania tokenów matematycznych.
