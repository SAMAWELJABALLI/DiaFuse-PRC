# DiaFuse-PRC — diagnostic de B12 face à B11 et protocole DiaFuse-PRC++

**Nature de ce document.** Tous les chiffres ci-dessous sont **lus ou recalculés à partir des résultats déjà archivés** :
- `results/summary.json` et `results/results.tex`, du run quick (3 seeds pour les systèmes principaux, 2 pour les ablations) ;
- `runs/dev_scores.json` ;
- `runs/diag/B12/*` ;
- le log du notebook exécuté.

**Aucune nouvelle expérience n'a été exécutée.** L'environnement n'a ni GPU, ni PyTorch, ni accès aux modèles ou aux données. Aucun chiffre de DiaFuse-PRC++ n'existe encore.

Les scripts de recalcul sont dans `pp_report/` :
- `mapping_check.py` → `table_mapping.csv` ;
- `failure_analysis.py` → `failure_analysis_output.md`.

---

## A. Mapping exact des résultats actuels

Le mapping a été reconstruit automatiquement selon la chaîne : clé `\res{système/set/métrique}` de chaque cellule dans `sec6results.tex` → valeur dans `results/results.tex` → définition dans `diafuse_eval.py` :
- `CHAIN = [ar, gs, gu, tl, tz, na, my]` correspond à S0…S6 ;
- `PAIR = {gs: ar, gu: ar, na: naar}` définit les paires de l'EM.

Le fichier complet est `pp_report/table_mapping.csv`.

| Table | Colonne | Scénario | Métrique | Clé | B11 | B12 |
|---|---|---|---|---|---|---|
| **Table 5** | Accuracy S0 | S0 TSAC-AR | accuracy, moyenne 3 seeds | `*/ar/acc:m` | **96.1** | **90.9** |
| Table 5 | Accuracy S3 | S3 TSAC-Latin | accuracy | `*/tl/acc:m` | **86.2** | **80.8** |
| Table 5 | Accuracy S6 | S6 MYC | accuracy | `*/my/acc:m` | **51.6** | **48.2** |
| Table 5 | EM (S0,S1) | paire ar–gs | exact match | `*/gs/em:m` | 91.9 | 85.8 |
| Table 5 | EM (S0,S2) | paire ar–gu | exact match | `*/gu/em:m` | 74.9 | 67.8 |
| Table 5 | EM (S5ar,S5) | paire naar–na | exact match | `*/na/em:m` | 68.1 | 61.6 |
| **Table 4** | S0…S6 | S0…S6 | macro-F1, moyenne ± sd sur 3 seeds | `*/<set>/mf1` | 96.1 / 91.8 / 72.2 / 86.1 / 69.9 / 71.0 / 51.6 | 90.8 / 84.4 / 59.2 / 80.6 / 65.3 / 70.3 / 48.2 |

**Correction confirmée.** Les valeurs 96.1 / 86.2 / 51.6 (B11) et 90.9 / 80.8 / 48.2 (B12) sont des **accuracies sur S0, S3 et S6**. Il ne s'agit ni de S0/S1/S2, ni de macro-F1.

La Table 5 ne contient **aucune** valeur pour S1 ou S2 en accuracy. Ces deux scénarios n'y apparaissent qu'à travers l'EM, qui mesure l'accord entre paires et non la justesse.

## B. Écart quantitatif B12 − B11

`*` indique |Δ| > le plus grand des deux écarts-types de seed. C'est le critère (ii) de la règle de décision du papier ; ce n'est pas un test de significativité.

| Métrique | S0 | S1 | S2 | S3 | S4 | S5 | S6 | moy. S0–S6 | moy. S3–S6 |
|---|---|---|---|---|---|---|---|---|---|
| macro-F1 | −5.3* | −7.4* | **−13.0*** | −5.5* | −4.6* | −0.7 | −3.4* | −5.69 | −3.53 |
| accuracy | −5.2* | −7.3* | −9.0* | −5.4* | −4.5* | −0.9 | −3.4* | −5.10 | −3.55 |

1. **Scénarios perdus** : S0, S1, S2, S3, S4 et S6, avec un écart supérieur au bruit de seed. Le plus grand écart est sur S2, la convention inconnue : −13.0.
2. **Scénario compétitif** : S5 (NArabizi), avec −0.7 en macro-F1, à l'intérieur du bruit (sd de B11 = 1.8). Sur S6, l'écart de −3.4 dépasse le bruit, mais les deux systèmes sont proches du hasard.
3. **Écart minimal** : sur les 7 scénarios, B12 n'est **jamais** au-dessus de B11.

## C. Diagnostic des ablations

### C1. Matrice des composants

Δ = modèle complet − modèle sans le composant, en points de macro-F1.
- **Référence appariée** : `fact-1111`. C'est exactement la configuration de B12 avec les seeds 0–1, qui sont aussi les seeds des ablations. Comparer à B12, qui a 3 seeds, serait biaisé.
- Δ > 0 signifie que le composant **aide**.
- Les ablations n'ont que **2 seeds** : `*` est un indicateur de bruit, pas une preuve.
- Pour chaque ligne, toutes les autres parties du modèle sont celles de B12, y compris la gate effondrée.

| Composant (avec vs sans) | S0 | S1 | S2 | S3 | S4 | S5 | S6 | Global | Hum S3–S6 | WGA | ECE_var |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Gate adaptative vs concat statique** | −0.6 | −2.6* | −0.0 | +4.5* | +1.6 | +4.2 | +2.9* | +1.40 | **+3.29** | +13.1 | −3.0 |
| 3 encodeurs vs meilleur seul (mE5) | −0.2 | −1.2 | −2.7* | −0.3* | −0.4 | −0.7 | +0.2 | −0.77 | −0.30 | −1.0 | +0.2 |
| **Descripteur de surface dans la gate** | +1.8* | +1.8 | −0.3 | +4.8* | +0.7* | +4.2 | −1.0 | +1.72 | **+2.17** | +12.5 | −0.8 |
| Mélange de couches appris vs dernière couche | −1.2* | +2.0* | **−5.4*** | +6.8* | −1.0* | +2.5* | −0.9 | +0.41 | +1.85 | +15.1 | −0.1 |
| **Prototypes (hiér.) vs aucun** | +0.2 | −0.2 | **−6.0*** | −0.6* | −0.7* | −1.7* | −2.1* | **−1.58** | **−1.26** | −2.0 | +2.9 |
| Prototypes hiér. vs classe seule | +1.3* | +0.2 | −2.2 | +0.4 | −0.6* | −0.1 | −1.0 | −0.29 | −0.33 | +0.3 | +0.4 |
| Hard mixup vs aucun | −0.2 | −1.5* | −2.5* | +1.8* | −0.4 | +2.8* | +1.1* | +0.17 | +1.34 | +3.3 | −1.5 |
| Dureté multi-signal vs prototype seul | −0.2 | +1.1 | −2.9* | +0.4 | +0.3 | +0.1 | −1.0* | −0.34 | −0.08 | −0.4 | +0.0 |
| **Consistance vs aucune** | −0.0 | +0.2 | −0.8 | +1.7* | −0.2 | −1.0 | +0.2 | **+0.01** | +0.17 | +4.6 | +0.0 |
| Calibration (MMCE + T_v) vs aucune | +1.0* | −0.6 | −0.6 | +1.0* | +0.6* | +0.6 | +1.0* | +0.43 | +0.80 | +4.4 | −0.1 |
| **Arabizi synthétique vs arabe seul** | +0.2 | +29.8* | +8.1* | +20.3* | +5.3* | +13.0* | +5.3* | **+11.72** | **+10.98** | +34.6 | −11.3 |
| κ=50 vs κ=0 | +0.3 | −0.5 | −3.0* | −0.3 | −0.6 | −1.3* | −0.4 | −0.83 | −0.65 | −0.8 | +0.9 |
| κ=50 vs κ=∞ | +0.8* | +0.8 | −2.5 | +0.5* | +0.2 | −0.8 | −0.5 | −0.21 | −0.14 | −0.1 | +0.9 |
| Pooling moyenne+CLS vs moyenne seule | +2.0 | +2.6 | −4.7* | +8.9* | +2.4* | +13.6* | +2.5* | +3.91 | +6.85 | +20.2 | −5.5 |
| Pooling moyenne+CLS vs CLS seul | +8.3* | +10.1* | +2.0 | +12.4* | +3.5 | +4.4 | +3.4* | +6.32 | +5.95 | +21.5 | +1.3 |
| Projection A partagée vs indépendante | +1.4* | +2.7* | −1.9* | +10.1* | +3.5* | +16.7* | +1.6* | +4.88 | +7.99 | +22.3 | −7.1 |
| Familles de consistance : toutes vs TR seul | −0.7 | −0.7 | **−6.8*** | +0.9* | −0.1 | −0.3 | +0.4 | −1.03 | +0.22 | +0.0 | +0.2 |

D'autres lignes figurent dans `pp_report/failure_analysis_output.md` : gate sur s seulement, M=2, termes de consistance, gate-consistency.

**Ce qui aide nettement :**
- les données synthétiques (+11.0 sur la chaîne humaine) ;
- les choix de pooling et de projection partagée (+6 à +8) ;
- la gate et le descripteur de surface (+2 à +3).

**Ce qui n'aide pas ou dégrade :**
- les prototypes : −1.3 sur la chaîne humaine, −6.0 sur S2 ;
- la consistance : ≈ 0 ;
- le passage de 3 à 1 encodeur ne change presque rien (−0.3), ce qui est cohérent avec l'effondrement de la gate.

**S2 est le scénario le plus fragile.** Les prototypes (−6.0), le mélange de couches (−5.4) et la consistance sur INF/NOISE/CS (−6.8) le dégradent tous. C'est aussi le scénario le plus bruité (sd de B12 = 2.5).

### C2. Décomposition de l'écart à B11 (chaîne de systèmes)

| Contraste | S0 | S1 | S2 | S3 | S4 | S5 | S6 | Global | Hum |
|---|---|---|---|---|---|---|---|---|---|
| **Fine-tuning** : B11 vs B3 (meilleur encodeur gelé, mE5 + MLP) | +4.8* | +9.2* | +4.0* | +5.1* | +4.6* | +0.1 | +0.2 | **+4.00** | **+2.51** |
| Fine-tuning, même famille d'encodeur : B11 vs XLM-R gelé + MLP (1 seed) | +11.3 | +12.5 | +15.2 | +16.7 | +7.5 | +10.3 | +2.4 | +10.84 | +9.22 |
| Fusion statique : B6 vs B3 | +0.7 | +2.4* | +1.0 | −1.4 | +1.6* | −3.0* | −1.3 | −0.00 | −1.04 |
| **DiaFuse vs concat statique** : B12 vs B6 | −1.2* | −0.6 | **−10.1*** | +1.1 | −1.6* | +2.5* | −1.9* | **−1.69** | +0.02 |
| Squelette DiaFuse sans auxiliaires (`fact-0000`) vs B6 | −0.9 | +1.1 | −10.1* | −5.3* | −2.9* | −3.8* | −3.7* | **−3.65** | **−3.93** |
| Auxiliaires + gate sur le squelette : `fact-1111` vs `fact-0000` | −0.5 | −1.8* | −1.4 | +6.4* | +1.3* | +6.1* | +2.0* | +1.74 | +3.96 |

**Lecture.** L'écart moyen de −5.69 points entre B12 et B11 se décompose approximativement ainsi :
- environ **4.0 points** viennent du fine-tuning de l'encodeur ;
- environ **0 point** vient de la fusion statique ;
- environ **1.7 point** vient d'un déficit propre à DiaFuse, concentré sur S2.

À l'intérieur de DiaFuse, le **squelette** (mélange de couches + chemin de décision linéaire) coûte −3.9 points sur la chaîne humaine par rapport à B6. Les auxiliaires et la gate en rattrapent +4.0.

**Borne supérieure en Setting A.** Même en choisissant, a posteriori et stage par stage, le meilleur des 12 systèmes gelés, on obtient 74.67 de moyenne sur S0–S6, contre 76.96 pour B11. Cette sélection oracle reste inférieure à B11 sur 6 des 7 stages. **Dans le cadre « encodeurs gelés », battre B11 sur plusieurs scénarios est donc peu plausible.**

### C3. Réponse à la question 6 : d'où vient le problème ?

**Plusieurs facteurs, par ordre d'importance mesurée :**
1. **L'absence d'adaptation des encodeurs** (représentation) : environ 4 points en moyenne, et 10.8 points à famille d'encodeur égale. Elle touche surtout S0, S1, S3 et S4.
2. **Le squelette de DiaFuse** (chemin linéaire + mélange de couches) : −3.9 points sur la chaîne humaine contre B6, −10 sur S2. Le mélange de couches et le chemin linéaire ne peuvent pas être séparés avec les runs existants.
3. **La gate effondrée** : la fusion multi-encodeurs n'apporte rien (−0.3 contre le seul mE5).
4. **Des auxiliaires mal pondérés** : prototypes nuisibles, consistance neutre, interactions négatives dans le factoriel.
5. **La calibration** n'explique pas les écarts de macro-F1 : effet +0.8, et la décision n'est pas affectée.

## D. Diagnostic de l'effondrement de la gate

**Constat.** Le poids de mE5 vaut 1.00 et l'entropie 0.00 pour **chaque** exemple de chaque set, sur les 3 seeds. C'est vérifié dans `runs/diag/B12/{0,1,2}/*.tsv` : 0 exemple avec α_mE5 < 0.99.

**Mécanisme, d'après le code (`diafuse_run.py`, `DiaFuse.forward` et `fit`) :**
1. α = softmax(g(x)/τ_g) avec τ_g = 1, et la dernière couche du MLP de gate n'est pas contrainte.
2. z = Σ α_m e_m. Le gradient sur le logit m vaut α_m⟨∂L/∂z, e_m − z⟩/τ. Quand α est presque one-hot, ce gradient tend vers 0 pour **tous** les m : l'état effondré est **absorbant**.
3. Une fois l'effondrement atteint, les branches XLM-R et LaBSE ne reçoivent plus de gradient de classification (z ne les contient plus). Leurs représentations ne s'améliorent pas : dynamique « rich-get-richer ».
4. La seule protection est λ_g · KL(ᾱ‖u) avec λ_g = 0.01. Sa valeur maximale, atteinte à l'effondrement, est 0.01 · ln 3 = **0.011**, négligeable devant la cross-entropie. Son gradient s'annule aussi quand α sature.
5. Il n'y a ni bruit, ni dropout d'encodeur, ni plancher, ni contrôle de température.
6. mE5 est le meilleur encodeur seul sur le dev (0.884 contre 0.858 pour LaBSE et 0.849 pour XLM-R) : c'est l'attracteur naturel.

**Inconnues (non mesurées dans le run archivé) :**
- l'époque de l'effondrement ;
- l'effondrement éventuel des variantes ablatées : le diagnostic n'a été écrit que pour B12. En particulier, on ne sait pas si `abl-nos` s'est effondré sur un autre encodeur.

**Ce que les données n'établissent pas.** La gate effondrée sur mE5 bat la concat statique linéaire (+3.3 sur la chaîne humaine). Cela montre seulement que « mE5 seul » vaut mieux que « concat linéaire » dans ce squelette. **Rien ne montre qu'une gate diversifiée généraliserait mieux.** C'est précisément l'hypothèse à tester sur Colab, et l'entropie seule ne suffira pas à la trancher.

**Test prévu** (chaîne causale « gate diversifiée → meilleure représentation → meilleure généralisation ») :
- (a) les variantes V1 mesurent sur le dev l'entropie, la part d'argmax par encodeur, la différence de routage AR/LAT, la norme de chaque e_m et l'époque d'effondrement ;
- (b) la sélection se fait sur le dev uniquement ;
- (c) au test, l'ablation pré-déclarée **FINAL-origgate** (même modèle avec la gate d'origine), sur 5 seeds, mesure l'effet de la correction sur S0–S6.

Un gain d'entropie sans gain de (c) sera rapporté comme tel.

## E. Hypothèses principales expliquant l'échec de B12

| # | Hypothèse | Preuve existante | Test sur Colab |
|---|---|---|---|
| E1 | **B11 bénéficie de l'adaptation de l'encodeur ; B12 est gelé.** Comparer les deux confond architecture et régime d'entraînement. | C2 : +4.0 global (B11 vs B3), +10.8 à famille égale ; oracle gelé < B11 sur 6/7 stages | Setting B (FT0–FT5) |
| E2 | **Chemin de décision linéaire.** LayerMix, mélange convexe et W sont linéaires : B12 est en pratique une sonde linéaire sur des features normalisées, alors que B6 a un MLP. | `fact-0000` vs B6 : −3.9 sur la chaîne humaine (confondu avec le mélange de couches) | V3 (têtes), V6 (entrées style B6) |
| E3 | **Effondrement absorbant de la gate** (section D). | Diagnostics B12 ; B12 ≈ `abl-single-me5` | V1, V2 + FINAL-origgate |
| E4 | **Sur-ajustement à la convention G_seen.** L'entraînement, les partenaires de consistance, le mixup et la sélection dev utilisent tous G_seen. | S2 : −10.1 contre B6 ; prototypes −6.0, couches −5.4, familles −6.8 sur S2 ; `abl2-twogen` S2 = 78.8 | V4, V5b, V6 (sélection sur dev) |
| E5 | **Auxiliaires non pondérés** : 5 pertes à 0.5, jamais réglées. | C1 : prototypes nuisibles, consistance neutre, interactions négatives | V5a, V4c |
| E6 | **Optimisation non réglée** : lr 1e-3 constant, sans warmup, et aucun tuning en mode quick. | Configuration du run | V5c |
| E7 | **Le dev est un mauvais prédicteur des stages humains** pour les variantes DiaFuse. | Spearman(dev, Hum) = −0.11 sur 46 variantes DiaFuse (`dev_scores.json` vs `results.tex`, **analyse post hoc**) | Risque pour toute sélection (cf. H) |

## F. Modifications proposées (DiaFuse-PRC++), par probabilité d'amélioration

1. **V5 — Setting B, adaptation des encodeurs (LoRA)**, avec gate corrigée, tête MLP et consistance ℓ2 + JS (FT2–FT5). C'est la seule voie qui attaque E1, le facteur dominant.
2. **V6 — Entrées style B6** (dernière couche, pooling par défaut) + objectif DiaFuse. Attaque E2 et E4 ; données : squelette −3.9, mélange de couches −5.4 sur S2.
3. **V3 — Tête non linéaire** (`mlp`, `ln_mlp`, `resmlp`, capacité d×d). Attaque E2.
4. **V1 — Corrections de la gate** :
   - initialisation nulle + τ=2 ;
   - λ_lb=0.1 ;
   - dropout d'encodeur 0.2 ;
   - τ appris + plancher 0.1 ;
   - bruit + entropie.
5. **V2 — Fusion** : gated residual, concat+MLP, attention entre encodeurs, somme pondérée globale. Toutes avec LayerNorm par encodeur.
6. **V4 — Prototypes** : supprimés, utilisés dans la décision, ou avec une perte allégée.
7. **V5a/b/c — Pertes et optimisation** :
   - auxiliaires allégés (0.1) ;
   - consistance limitée à TR ;
   - cosine + warmup avec lr 5e-4.

Deux variantes (V4a, V5b) sont **motivées a posteriori** par des ablations évaluées sur le test. Elles restent des candidates sélectionnées **sur le dev uniquement**, et cette motivation est signalée.

Les combinaisons ne sont construites qu'à partir des **gagnants de famille** : leur union, plus l'union privée de chaque famille. Il n'y a pas d'explosion combinatoire.

## G. Protocole expérimental (Colab A100)

| Phase | Contenu | Test lu ? |
|---|---|---|
| 0 | Téléchargement, `prep_data.py`, **vérification que les splits sont identiques à ceux du run archivé** (sinon arrêt), extraction des features | non |
| 1 | V0 (code original) et P0 (nouvelle implémentation, valeurs par défaut) sur 3 seeds dev, comparés à `archived_dev_scores.json` ; B11 seeds 3–9 avec `finetune.py` inchangé | non (B11 : prédictions écrites comme dans le protocole d'origine) |
| 2–3 A | 22 configurations × 3 seeds (`search` : dev seulement, avec log par époque de la gate et des normes) ; `select screen` ; combinaisons × 3 seeds ; `select final` | **non** |
| 4 A | FINAL × 10 seeds ; V0 complété à 10 seeds ; ablations pré-déclarées × 5 seeds | **oui, une seule fois** |
| 5 B | FT0 (B11 ré-implémenté, contrôle d'implémentation), FT1 (contrôle), FT2–FT5 × 3 seeds (dev) → sélection → FINAL × 10 seeds + ablations × 5 seeds | dev, puis une seule fois |
| 6 | `diafuse_eval.py` inchangé (sortie dans `results_v2/`, l'archive n'est pas écrasée) + `compare.py` | une seule fois |

`compare.py` produit :
- le tableau S0–S6 (moyenne ± sd, IC t sur les seeds, IC bootstrap sur les exemples) ;
- les Δ par rapport à B11 : moyenne, médiane, nombre de Δ > 0 / > 1 / > 2 ;
- un bootstrap à deux niveaux (seeds × exemples) et un test de Welch sur les seeds, avec correction de Holm sur les 7 stages ;
- le bloc de la Table 5.

## H. Règle de sélection exacte (fixée avant tout run)

1. **Score(config, seed)** = moyenne du macro-F1 sur **val_ar et val_gs**, à l'époque retenue par early stopping. C'est le critère du protocole, utilisé aussi pour B11 et toutes les baselines. Poids : ½ chacun. Aucun scénario S0–S6 n'intervient.
2. **Score(config)** = moyenne sur les seeds 0, 1, 2 (au moins 3 exigés).
3. **Shortlist** : une configuration y entre si Score(config) − Score(référence) > max(sd_config, sd_référence). La référence est V0 en Setting A et FT0 (recette B11) en Setting B.
4. **Gagnant de famille** = meilleur Score parmi les configurations shortlistées de la famille. Combinaisons : ALL et ALL-minus-<famille>, chacune sur 3 seeds.
5. **FINAL** = meilleur Score parmi les gagnants et les combinaisons qui passent l'étape 3.
   - En cas d'égalité (écart < 0.2 point), on prend le modèle avec le moins de paramètres entraînables.
   - **Si rien ne passe, FINAL = référence**, et le résultat est rapporté comme négatif.
6. FINAL et les ablations pré-déclarées sont entraînés **une fois** sur les seeds 0–9 (ablations : 0–4), puis évalués **une fois**. Aucune re-sélection après lecture du test.
   - Ablations en Setting A : FINAL-minus-{cons, cal, hm, proto}, FINAL-origgate, ALL-minus-<famille>.
   - Ablations en Setting B : FINAL-static, FINAL-nojs.

**Pourquoi ne pas utiliser un dev composite « robuste » ?**
- **(a) Symétrie** : B11 et toutes les baselines ont été sélectionnés sur val_ar + val_gs.
- **(b) Absence de proxy meilleur.** Une analyse post hoc sur les runs existants le montre : sur 78 systèmes, la corrélation de Spearman avec la moyenne humaine vaut 0.40 pour S0+S1, et 0.39 pour un proxy incluant les perturbations INF/NOISE/CS. Seul le proxy fondé sur G_unseen corrèle mieux (0.61), et il est interdit puisque S2 doit rester inconnu.
- Cette observation utilise des résultats de test **existants** et a servi **uniquement** à décider de **ne pas** modifier le critère du protocole. Elle est signalée comme telle.
- **Conséquence assumée (E7)** : la sélection dev peut ne pas retenir la meilleure configuration sur les stages humains.

## I. Coût de calcul estimé (A100-40GB)

Les estimations sont extrapolées des durées du log archivé :
- B12 : environ 6 min par seed, diagnostic compris ;
- B11 : 11.6 min par seed ;
- les durées LoRA à 3 encodeurs **ne sont pas mesurées** (fourchette).

| Bloc | Runs | Durée estimée |
|---|---|---|
| Phase 0 (modèles, données, features) | — | ~0.5 h |
| Phase 1 (V0, P0 × 3) + B11 seeds 3–9 | 6 + 7 | ~0.6 h + ~1.4 h |
| Recherche A (22 × 3) + combinaisons (≤ 7 × 3) | ≤ 87 | ~9 h |
| Final A (10 + 7 B12 + ≤ 9 ablations × 5) | ≤ 62 | ~6.5 h |
| Recherche B : FT0, FT1 (full FT, ~12 min) ; FT2–FT5 (LoRA ×3, 25–40 min, non mesuré) × 3 | 18 | ~6–9 h |
| Final B (10 + 2 ablations × 5) | 20 | ~8–13 h |
| Évaluation + compare | — | ~0.5 h |
| **Total** | | **~32–40 h GPU** |

**Version prioritaire (~14 h)**, si le budget est limité :
- phases 0 et 1 ;
- Setting B FT0, FT2 et FT3 × 3 seeds ;
- Setting A V1a, V1c, V3a, V6a, V6b × 3 seeds ;
- puis les runs finaux sur 5 seeds au lieu de 10.

Cette réduction est à déclarer **avant** de lancer, et elle sera rapportée.

## J. Fichiers à exécuter sur Colab

Dossier **`DiaFuse-PRC-pp/`**, à copier sur Drive (`/content/drive/MyDrive/DiaFuse-PRC-pp`) :

| Fichier | Rôle |
|---|---|
| `DiaFuse-PRC-pp_Colab.ipynb` | Orchestration complète, phases 0 à 6 (drapeaux `RUN[...]`) |
| `orig_code/*` + `MD5SUMS_unmodified.txt` | Scripts originaux **inchangés** (vérifiés par MD5 dans le notebook) |
| `pp/diafuse_pp.py` | Setting A : V0 = trainer original ; autres configurations = `PPModel` (gate, fusion, tête, prototypes, pertes, entrées) ; modes `search` (dev seulement) et `final` |
| `pp/diafuse_pp_ft.py` | Setting B : LoRA ou fine-tuning multi-encodeurs, recette B11 + objectif DiaFuse ; modes `search` et `final` |
| `pp/select_config.py` | Règle de sélection H (ne lit que `runs_pp/dev/`) |
| `pp/gate_report.py` | Comportement de la gate sur le dev (avant / après) |
| `pp/check_repro.py` | Contrôles de reproduction (scores dev, accord des prédictions) |
| `pp/compare.py` | Tableaux finaux D, E et Table 5, avec statistiques |
| `configs/phase3_settingA.json`, `configs/phase3_settingB.json` | Configurations, avec la justification (`why`) de chacune |
| `configs/archived_prep_report.json`, `configs/archived_dev_scores.json` | Références de reproduction |

**Limite de validation du code.** Les scripts ont été compilés et vérifiés statiquement ici (`py_compile`, noms non définis). `select_config.py` et `compare.py` ont été exécutés sur des fichiers factices.

Les scripts PyTorch (`diafuse_pp.py`, `diafuse_pp_ft.py`) **n'ont pas pu être exécutés**, faute de PyTorch dans cet environnement. Les phases 0 et 1 servent précisément de tests :
- **P0 doit reproduire V0 sur le dev.** L'ordre d'initialisation et la consommation des générateurs aléatoires ont été conservés, donc l'égalité est attendue au non-déterminisme GPU près. Un écart supérieur au bruit de seed signale un défaut de la nouvelle implémentation, et il faut alors s'arrêter.
- **FT0 doit reproduire B11 sur le dev** (0.9400, 0.9351, 0.9358) à la variabilité de seed près. Il ne peut pas être identique au bit près : les têtes de projection supplémentaires changent la séquence aléatoire de PyTorch.
