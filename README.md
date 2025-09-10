## CLI Commands

```bash
# List sites
python app/cli.py sites

# Forecast for site (default 14 days)
python app/cli.py forecast <site> [--days N] [--format table|json|csv] [--force-retrain]

# Anomalies for site
python app/cli.py anomalies <site> [--start-date DD-MM-YYYY] [--end-date DD-MM-YYYY] [--severity low|medium|high] [--format table|json|csv] [--force-retrain]

# Site overview (forecasts + recent anomalies)
python app/cli.py overview <site> [--days N] [--recent-days N] [--format table|json] [--force-retrain]

# Train all models
python app/cli.py train-all [--force-retrain]

# Check pipeline health
python app/cli.py health
```
## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python app/cli.py forecast S1
```
