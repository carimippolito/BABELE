# BABELE

Progetto per il corso di Fondamenti di Visione Artificiale e Biometria e il corso di Software Engineering for Artificial Intelligence (Dipartimento di Informatica @UNISA, anno accademico 2021/2022).

## Introduzione
Lo scopo del progetto è quello di realizzare un sistema di riconoscimento che utilizzi video contenenti solo l'area labiale, e valutare se conoscere la lingua del soggetto (che deve essere identificata) permetta di ottenere risultati migliori. Il progetto contiene anche un'analisi iniziale su alcuni dei requisiti non funzionali (fairness ed explainability).

---

#### [Codice](../main/Codice):

    Data acquisition: script e risultati della fase di acquisizione dei dati.

    Data understanding: script e risultati della fase di comprensione dei dati.

    Modeling: script per l'identificazione della lingua.

    Modeling2: script per il riconoscimento.

    Prototyping: script contenenti i sistemi multimodali.

#### [Dataset (identificazione della lingua)](../main/Dataset):

    Csv_curvatura_di_Ricci: risultati degli esperimenti sulla curvatura di Ollivier-Ricci.

    Csv_distanza_di_X: distanze utilizzate per l'identificazione della lingua.

#### [Dataset2 (riconoscimento)](../main/Dataset2):

    Csv_distanza_di_X: distanze utilizzate per il riconoscimento.

    Csv_immagini: immagini utilizzate per il riconoscimento.

#### [mlruns.7z](../main/mlruns.7z): risultati dei parametri testati dalla grid search.

#### [environment.yml](../main/environment.yml): dipendenze utilizzate all'interno del progetto.
