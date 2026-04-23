# Summary Method Comparator

Compares 10 summarization methods side by side.

## Setup

```bash
# Clone
git clone https://github.com/yourusername/summary-comparator
cd summary-comparator

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```


## Methods
- LSA, LexRank, TextRank, Luhn, SumBasic, KL (Extractive)
- T5-Small, BART (Abstractive)

## Note
Models will be downloaded automatically on first run:
- T5-Small (~60MB)
- BART (~1.6GB)