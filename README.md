# AI Financial Analyst

This project was developed as part of the MSc Data Analytics programme at the National College of Ireland for the Deep Learning & Generative AI module. The objective was to investigate whether Retrieval-Augmented Generation (RAG) using SEC 10-K filings improves the quality of AI-generated financial analysis compared with conventional prompting techniques.

The project builds an automated financial analyst that combines structured financial ratios with text from real SEC 10-K filings to produce analysis reports, and compares how much retrieval actually helps compared to prompting alone.

## Table of Contents

- [What this project does](#what-this-project-does)
- [Team](#team)
- [Repository structure](#repository-structure)
- [How the pipeline fits together](#how-the-pipeline-fits-together)
- [Notebooks](#notebooks)
- [Outputs](#outputs)
- [Setup](#setup)
- [Running the project](#running-the-project)
- [Technologies used](#technologies-used)
- [Limitations](#limitations)
- [Possible improvements](#possible-improvements)

## What this project does

The project has two parts.

The first is a fairly standard classification problem: given a company's financial ratios for a given year, predict whether the stock is a Buy, Hold, or Sell for the following year. This is handled with a small MLP trained on cleaned financial data.

The second part is the main focus of the project. We use an LLM (Google Gemini) to generate three types of reports for each company: an executive summary, a risk assessment, and an investment recommendation. Each report is generated four different ways: with no extra context (zero-shot), with one example added (few-shot), with a role instruction telling the model it's a senior analyst, and with retrieved text from the company's actual 10-K filing added to the prompt (RAG). The point of doing it four ways is to see whether adding retrieved filing text actually changes the quality of the output, rather than just assuming it does.

Everything is built around 22 companies, spanning 11 sectors, over fiscal years 2014–2018 (110 company-year combinations in total). These companies weren't picked at random from the dataset. They were checked against the live SEC EDGAR API to confirm each one actually files 10-Ks and has full coverage across all five years. This validation required multiple iterations because several initially selected companies did not have complete filing coverage across the study period (see the Member 1 notebook for the rounds of swapping companies in and out).

## Team

- Member 1: financial data cleaning, company selection against SEC EDGAR, feature engineering, Buy/Hold/Sell labelling, MLP baseline model.
- Member 2: knowledge base and EDGAR corpus validation, narrative extraction, text chunking, embeddings and FAISS index, RAG report generation, prompt design, generation performance analysis.
- Member 3: merging all outputs into one evaluation dataset, automatic evaluation metrics, and manual/qualitative comparison of the generated reports.

## Repository structure

```
AI_FINANCIAL_ANALYST/
│
├── data/
│   ├── raw/                 # Member 1's original yearly financial CSVs (2014–2018)
│   ├── edgar/                # Historical SEC EDGAR 10-K filings (JSON, by year)
│   └── processed/            # Cleaned data and everything produced by Notebooks 1–6
│
├── data_processed/           # Member 1's own working folder, mirrors data/processed
│   └── checkpoints/           # Intermediate CSVs saved at each cleaning/feature step
│
├── eda/                      # Saved plots (sector distribution, confusion matrix, etc.)
│
├── mlp_baseline/              # MLP predictions on the test set
│   ├── predictions.csv
│   └── predictions_by_company.csv
│
├── models/
│   └── mlp_baseline.keras
│
├── notebooks/
│   ├── Member1_Data_Preprocessing.ipynb
│   ├── Member2_01_Knowledge_Base_Setup.ipynb
│   ├── Member2_02_EDGAR_Corpus_Validation.ipynb
│   ├── Member2_03_Narrative_Extraction.ipynb
│   ├── Member2_04_Text_Preprocessing.ipynb
│   ├── Member2_05_Embeddings_FAISS.ipynb
│   ├── Member2_06_RAG_Report_Generation.ipynb
│   └── Member3_Evaluation.ipynb
│
├── .env                       # GEMINI_API_KEY, not committed (see .gitignore)
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## How the pipeline fits together

```
                          Financial Data
                                │
                                ▼
                          MLP Baseline
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             Structured Data          SEC 10-K Narratives
                                              │
                                              ▼
                                       Text Processing
                                              │
                                              ▼
                                     Sentence Embeddings
                                              │
                                              ▼
                                         FAISS Index
                                              │
                                              ▼
                                  Company-Year Retrieval
                                              │
                                              ▼
                                   Prompt Construction
                                              │
                                              ▼
                                       Google Gemini
                                              │
                                              ▼
                                   Executive Summary
                                   Risk Assessment
                                   Investment Recommendation
```

Member 1's notebook comes first and produces three files that everything else depends on: `company_mapping.csv`, `processed_financial_data.csv`, and `selected_companies.csv`. From there, Member 2's six notebooks run in order, each one picking up where the last left off:

1. Check that Member 1's data is complete and consistent.
2. Check that the EDGAR filing corpus is readable and has the sections we need.
3. Match each company-year to its filing and pull out the relevant text.
4. Clean that text and split it into chunks.
5. Turn the chunks into embeddings and build a FAISS index.
6. Use all of the above to generate reports with Gemini, across four prompting conditions.

Member 3's notebook then takes the final output of Notebook 6, merges it with the structured data, and evaluates it.

Nothing is skipped or reordered here. The notebooks genuinely need to run in this sequence because each one reads files written by the previous one.

## Notebooks

`Member1_Data_Preprocessing.ipynb`
Loads the five yearly CSVs, fixes a handful of duplicate columns caused by inconsistent capitalisation, drops columns that were missing more than 40% of their values, and imputes what's left using sector-level medians. Growth-rate columns get IQR clipping and large dollar-value columns get a signed log transform, since both had a lot of extreme outliers before this. Company selection was done against the live SEC `company_tickers.json` and `submissions` endpoints. A first pass of 24 candidates had several companies swapped out after we found they either don't file 10-Ks (foreign filers using 20-F/40-F instead) or didn't have full 2014–2018 coverage. Labels are assigned using a ±10% price-variation threshold. The MLP itself is a small three-layer dense network (64 → 32 → 16 units) with dropout and class weighting, trained on the pooled five years of data.

Notebook 1 – Knowledge Base Setup
Loads Member 1's three output files and checks them before anything else happens: column names, duplicates, missing values, that every ticker maps to exactly one CIK, and that every company has all five years present. This is basically a sanity check step before touching the EDGAR data.

Notebook 2 – EDGAR Corpus Validation
Looks at the raw EDGAR JSON corpus on disk (not the live API this time, but a local historical dataset) and confirms the folder structure, file counts per year, and that a sample filing actually contains `section_1`, `section_1A`, and `section_7`.

Notebook 3 – Narrative Extraction
Matches each row of `company_mapping.csv` to its filing using CIK and year, then pulls out Section 1 (Business), Section 1A (Risk Factors), and Section 7 (MD&A). Out of 110 company-years, 109 matched. The one exception is APTV 2017, which genuinely isn't in the historical EDGAR dataset we're using, even though the filing exists on SEC's live site. Rather than trying to work around it, it's just excluded from anything narrative-related going forward.

Notebook 4 – Text Preprocessing
Cleans up whitespace and formatting left over from the raw filings, then splits the text into chunks of 300 words with a 50-word overlap. The overlap is there so a sentence that gets cut in half at a chunk boundary still has some surrounding context in at least one chunk.

Notebook 5 – Embeddings & FAISS
Embeds every chunk using `all-MiniLM-L6-v2` from sentence-transformers and builds a FAISS `IndexFlatL2` index over them. This model was picked mainly because it's small and fast enough to run without a GPU, which mattered given the size of this project.

Notebook 6 – RAG Report Generation
The main notebook. For every company-year, it generates three reports (executive summary, risk assessment, investment recommendation) under four conditions (zero-shot, few-shot, role, RAG). All four conditions get the same structured financial numbers; the only thing that changes is whether retrieved filing text is added. For RAG, a small FAISS index is rebuilt on the fly from just that company's chunks each time, so there's no risk of retrieving text from a different company by mistake. For the risk assessment task specifically, retrieval is restricted to Section 1A only, with no fallback to another section if it's missing; five company-years don't have a Section 1A, so RAG is skipped for risk assessment on those, but the other three conditions still run normally.

Reports are generated using Gemini's free tier (`gemini-flash-lite-latest`), which comes with daily and per-minute limits. Because of that, the whole generation loop was written to be resumable: it saves progress every 25 records and can be stopped and restarted across multiple days without losing anything or duplicating work. The final cleaned output is 1,312 report records. The notebook ends with a short analysis of latency, token usage, and how many generations succeeded/failed/were skipped per condition.

`Member3_Evaluation.ipynb`
Starts by loading and checking all four datasets from Members 1 and 2, then merges them into one evaluation table keyed on ticker and year. From there it looks at generation success rates, response length, token usage and latency across the four prompting conditions, plus a more manual/qualitative pass comparing report quality between conditions.

## Outputs

| Output | Description |
|---|---|
| Processed financial dataset | Cleaned structured data used by the MLP (`processed_financial_data.csv`) |
| Company mapping | Ticker, CIK, sector, and company name for each company-year (`company_mapping.csv`) |
| Selected companies | Final 22-company shortlist, validated against SEC EDGAR (`selected_companies.csv`) |
| MLP predictions | Buy/Hold/Sell predictions and class probabilities on the test set (`mlp_baseline/predictions.csv`, `predictions_by_company.csv`) |
| Trained MLP model | Saved Keras model (`models/mlp_baseline.keras`) |
| Narrative sections | Extracted Section 1, 1A, and 7 text per company-year (`narrative_sections.csv` / `.parquet`) |
| Processed chunks | 300-word text chunks used for retrieval (`processed_chunks.csv` / `.parquet`) |
| Embeddings | SentenceTransformer vector representations of each chunk (`chunk_embeddings.npy`) |
| FAISS index | Vector database used during RAG retrieval (`faiss_index.bin`) |
| Chunk metadata | Ticker/year/section lookup for each embedded chunk (`chunk_metadata.parquet`) |
| Generated reports | Executive Summary, Risk Assessment, and Investment Recommendation reports across all four prompting conditions (`generated_reports_final.csv` / `.parquet`) |
| EDA plots | Sector distribution, label distribution, confusion matrix, and generation performance charts (`eda/*.png`) |

## Setup

Python 3.11 or later is recommended.

```bash
git clone https://github.com/rakhidesale/ai-financial-analyst.git
cd ai-financial-analyst

python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with your own Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

The key is loaded through `python-dotenv` and is never written directly into a notebook. `.env` is in `.gitignore` for this reason. It should not end up in the repo.

## Running the project

Run the notebooks in the following order:

1. `Member1_Data_Preprocessing.ipynb`
2. `Member2_01_Knowledge_Base_Setup.ipynb`
3. `Member2_02_EDGAR_Corpus_Validation.ipynb`
4. `Member2_03_Narrative_Extraction.ipynb`
5. `Member2_04_Text_Preprocessing.ipynb`
6. `Member2_05_Embeddings_FAISS.ipynb`
7. `Member2_06_RAG_Report_Generation.ipynb`
8. `Member3_Evaluation.ipynb`

Each notebook reads the files produced by the one before it, so this order is required, not just a suggestion. Step 7 needs a working `GEMINI_API_KEY` and calls a live API; because of the free-tier daily limit, it may need to be run over more than one session.

Or run the whole thing with:

```bash
python main.py
```

## Technologies used

- Python for everything
- pandas / NumPy / PyArrow for data handling (Parquet for the larger intermediate files)
- scikit-learn for preprocessing, splitting, and class-weight calculation
- TensorFlow / Keras for the MLP baseline
- sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- FAISS for vector search
- Google Gemini (`gemini-flash-lite-latest`), accessed through the OpenAI-compatible client
- python-dotenv to keep the API key out of the notebooks
- Matplotlib / Seaborn for the plots in `eda/`
- Jupyter for all notebooks
- SEC EDGAR (company tickers API, submissions API, and a local historical filing corpus) as the data source

## Limitations

- APTV's 2017 filing isn't in the EDGAR corpus we used, so that company-year is missing from all narrative/RAG analysis.
- Five company-years have no Section 1A (Risk Factors) text, so RAG-based risk assessment is skipped for those specifically. We deliberately didn't fall back to a different section for these, since that would change what the RAG condition is actually being evaluated on.
- Report generation relies on Gemini's free tier, which has daily and per-minute request limits. This is the main reason the generation notebook had to be built to resume across sessions rather than running straight through.
- The EDGAR corpus used here is a fixed local snapshot covering 2014–2018 only.
- 22 companies is a small sample. The comparison between prompting strategies is useful as a case study, but shouldn't be read as a statistically strong result.
- Even at a low temperature (0.2), the LLM doesn't produce identical output on repeated runs, which is expected but worth noting.

## Possible improvements

- Run the same pipeline on more companies and more recent years to see if the findings hold up.
- Add an explicit `Company: / Fiscal Year:` field near the start of each prompt. This was suggested during peer review but wasn't implemented once report generation was already underway, since changing the wording partway through would break the fairness of the four-condition comparison.
- Try a fallback retrieval strategy for the company-years missing Section 1A, evaluated as its own separate condition rather than mixed into the main RAG results.
- Add reference-based or LLM-as-judge scoring to the evaluation, alongside the length/latency/token metrics already used.
- Try a larger embedding model, or add re-ranking after the initial FAISS retrieval, and see if it changes report quality.
