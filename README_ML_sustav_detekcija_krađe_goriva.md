# ML sustav za detekciju krađe goriva

Projekt strojskog učenja za klasifikaciju događaja vezanih uz potrošnju goriva na temelju telemetrijskih podataka vozila.

## Cilj

Cilj projekta je klasificirati događaje u tri kategorije:

- krađa goriva
- normalna potrošnja
- točenje goriva

Skup podataka sadrži približno 1,2 milijuna zapisa s informacijama o razini goriva, brzini, GPS koordinatama, vremenu i drugim telemetrijskim značajkama.

## Obrada podataka

Projekt uključuje:

- analizu kvalitete podataka i nedostajućih vrijednosti
- analizu duplikata i stršila
- imputaciju nedostajućih vrijednosti
- konstrukciju vremenskih i domen-specifičnih značajki
- značajke temeljene na promjenama razine goriva i kliznim prozorima
- balansiranje klasa pomoću SMOTE-a

## Modeliranje

Uspoređeno je više pristupa klasifikaciji, uključujući:

- Random Forest
- XGBoost
- Naive Bayes
- neuronske mreže za tablične podatke
- Stacked XGBoost + MLP
- Stacked Random Forest + MLP
- Deep & Cross Network

Modeli su uspoređivani pomoću accuracy, Macro F1, Weighted F1, precision i recall metrike, uz poseban fokus na prepoznavanje krađe goriva.

Projekt uključuje i analizu pogrešno klasificiranih uzoraka te analizu utjecaja značajki na predviđanja.

## Tehnologije

**Python · Pandas · NumPy · scikit-learn · XGBoost · PyTorch · SMOTE · Matplotlib**

## Struktura

Glavni eksperiment i analiza nalaze se u Jupyter notebooku:

`ML_sustav_za_detekciju_krađe_goriva.ipynb`

## Autor

**Jakov Kordić**
