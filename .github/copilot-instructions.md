# Project Instructions

## Project Purpose

This repository contains a City Hospital consultation-data preprocessing and analysis project. `Main.py` is the end-to-end Python pipeline: it loads the source workbook, cleans visit-level data, builds a patient-level analytic base table (ABT), generates plots, and produces PDF reports and slides. `FinalProject.ipynb` and `new_simplificado.ipynb` are notebook-based analysis artifacts.

## Data and Outputs

- Treat the source workbook and source-level CSVs as inputs. Never modify or replace raw data unless explicitly asked.
- `ABT_city_hospital.csv`, `ABT_city_hospital_notebook.csv`, `Data_Dictionary_city_hospital.csv`, and `Data_Quality_Log.csv` are project data artifacts. Preserve their schemas and document intentional changes.
- Running `Main.py` writes or overwrites the ABT, data dictionary, six `plot*.png` files, `Report.pdf`, and `Slides.pdf` in the current working directory. Do not run the full pipeline just to inspect code; generated artifacts may be user-maintained.
- Keep cleaning logic, the data quality log, the data dictionary, notebook analysis, and report descriptions consistent. Where they disagree, identify the discrepancy rather than silently choosing a methodology.
- Do not expose or add individual-level health or financial data to examples, logs, or documentation.

## Development Guidance

- Preserve the existing pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and ReportLab approach unless a requested change calls for a different tool.
- Keep preprocessing deterministic and patient-level aggregation keyed by patient identifier. Maintain the ABT's one-row-per-patient contract.
- Prefer focused edits and validate the affected transformation or output shape without rewriting unrelated deliverables.
- There is no dependency manifest or test suite at present. Check available dependencies and use focused validation; the full pipeline requires the source workbook and writes deliverables to the working directory.
- Run the pipeline from the repository root so its relative input and output paths resolve as intended.