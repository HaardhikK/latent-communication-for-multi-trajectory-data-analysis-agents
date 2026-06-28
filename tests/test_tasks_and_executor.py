from __future__ import annotations

import json

from latent_agent.executor import execute_python_code
from latent_agent.tasks import TASKS


def test_clean_summary_scorer_accepts_correct_script(tmp_path):
    task = TASKS["clean_summary"]
    task.setup(tmp_path)
    code = """
import json
import pandas as pd
df = pd.read_csv('input.csv')
df = df.drop_duplicates()
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['units'] = pd.to_numeric(df['units'], errors='coerce')
df = df.dropna(subset=['revenue'])
by_city = df.groupby('city')['revenue'].sum()
out = {
  'rows_after_cleaning': int(len(df)),
  'total_revenue': float(df['revenue'].sum()),
  'mean_units': float(df['units'].mean()),
  'top_city': str(by_city.idxmax()),
}
open('summary.json', 'w', encoding='utf-8').write(json.dumps(out))
"""
    result = execute_python_code(code, tmp_path, attempt=1)
    score = task.score(tmp_path)
    assert result.succeeded
    assert score.passed


def test_linear_regression_scorer_accepts_correct_script(tmp_path):
    task = TASKS["linear_regression"]
    task.setup(tmp_path)
    code = """
import pandas as pd
from sklearn.linear_model import LinearRegression
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
model = LinearRegression().fit(train[['x']], train['y'])
out = pd.DataFrame({'id': test['id'], 'y_pred': model.predict(test[['x']])})
out.to_csv('predictions.csv', index=False)
"""
    result = execute_python_code(code, tmp_path, attempt=1)
    score = task.score(tmp_path)
    assert result.succeeded
    assert score.passed


def test_grouped_sales_scorer_accepts_correct_script(tmp_path):
    task = TASKS["grouped_sales"]
    task.setup(tmp_path)
    code = """
import pandas as pd
df = pd.read_csv('sales.csv')
out = df.groupby('region').agg(total_revenue=('revenue', 'sum'), row_count=('revenue', 'size')).reset_index()
out = out.sort_values('region')
out.to_csv('region_summary.csv', index=False)
"""
    result = execute_python_code(code, tmp_path, attempt=1)
    score = task.score(tmp_path)
    assert result.succeeded
    assert score.passed


def test_clean_summary_scorer_rejects_missing_output(tmp_path):
    task = TASKS["clean_summary"]
    task.setup(tmp_path)
    score = task.score(tmp_path)
    assert not score.passed
    assert score.score == 0.0
    assert "summary.json" in score.message


def test_clean_summary_scorer_rejects_non_object_json(tmp_path):
    task = TASKS["clean_summary"]
    task.setup(tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps([{"rows_after_cleaning": 3}]), encoding="utf-8")
    score = task.score(tmp_path)
    assert not score.passed
    assert score.score == 0.0
    assert "JSON object" in score.message
