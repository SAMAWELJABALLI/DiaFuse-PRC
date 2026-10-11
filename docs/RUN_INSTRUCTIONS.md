# How to run DiaFuse-PRC++ (no prior knowledge needed)

You need a Google account with Google Colab access to an **A100 GPU**, and several GB of free space in Google Drive (the three encoders are cached there).

## 1. Get the repository

**Option A, from GitHub.** Use this when the project owner has given you the repository link.
1. Go to <https://colab.research.google.com>, then *File → Open notebook → GitHub*.
2. Paste the repository link and open `notebooks/DiaFuse_PRCPP_ONE_CLICK.ipynb`.

The notebook downloads the rest of the repository by itself.

**Option B, through Google Drive.** Use this if option A does not work, for example with a private repository.
1. On the GitHub page of the repository, click *Code → Download ZIP*, and unzip the file on your computer.
2. Upload the unzipped folder, which contains `notebooks/`, `scripts/`, `src/` and `original/`, into **My Drive**.
3. In Google Drive, double-click `notebooks/DiaFuse_PRCPP_ONE_CLICK.ipynb` and choose *Open with → Google Colaboratory*.

## 2. Enable the GPU

In Colab: *Runtime → Change runtime type → Hardware accelerator: A100 GPU → Save*.

## 3. Run

*Runtime → Run all*. When Colab asks for access to Google Drive, accept: all results are saved there.

## 4. Wait

The full run takes more than one day of GPU time, over several Colab sessions.

If Colab disconnects or stops, open the notebook again and choose *Runtime → Run all* again. The run continues where it stopped, and finished experiments are not recomputed.

## 5. Where the results are

Everything is in **My Drive/DiaFuse_PRCPP_RUN/**:
- `results/`: one folder per model (`B11`, `B12_setting_A`, `B12pp_setting_A`, `B12_setting_B_lora`);
- `tables/`, `figures/`, `logs/`, `checkpoints/`;
- `FINAL_REPORT.md`;
- `final_archive/`: the archives to send.

## 6. Did it succeed?

The last lines of the notebook show one of these:

| Message | Meaning |
|---|---|
| **RUN COMPLETED** | Everything finished. |
| **RUN INCOMPLETE** | Some experiments failed. Run all cells once more; if the message stays, send the files below. |
| **FATAL ERROR** | A safety check stopped the run, for example because data or reproduction do not match. Do not retry: send the files below. |
| **RUN INTERRUPTED** | The session stopped. Run all cells again. |

## 7. What to send back

- **RUN COMPLETED**: send `My Drive/DiaFuse_PRCPP_RUN/final_archive/DiaFuse_PRCPP_FINAL_RESULTS_<date>.zip`. Send `DiaFuse_PRCPP_CHECKPOINTS_<date>.zip` only if asked.
- **RUN INCOMPLETE** or **FATAL ERROR**: share the whole folder `My Drive/DiaFuse_PRCPP_RUN`.

## 8. Please do not

- change the notebook, its parameters or any file of the repository;
- edit, rename or delete anything inside `DiaFuse_PRCPP_RUN`;
- run individual scripts by hand.
