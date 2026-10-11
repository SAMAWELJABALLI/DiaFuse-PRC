# Data (not stored in git)

The pipeline rebuilds every split itself; nothing has to be placed here. `original/code/download_models.py` fetches the public releases from GitHub, and `original/code/prep_data.py` rebuilds the splits. On Colab they are written to `My Drive/DiaFuse_PRCPP_RUN/data`.

| Corpus | Source repository (fetched by `download_models.py`) | Use | Status in this repository |
|---|---|---|---|
| TSAC | `fbougares/TSAC` | training, development, S0–S3 | not redistributed |
| TUNIZI | `chaymafourati/TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset` | S4 | not redistributed |
| NArabizi | `SamiaTouileb/Narabizi` | S5, S5AR | not redistributed |
| MYC | `MouadJb/MYC` | S6, S6AR | not redistributed |

The redistribution terms of these corpora have not been checked for this repository, so no raw or derived text is committed. The Latin-script views S1, S2 and S6SYN are generated from the corpora by `original/code/translit.py`.
