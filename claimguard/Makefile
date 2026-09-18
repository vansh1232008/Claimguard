.PHONY: help install dataset seed api web demo train graphsage evaluate test clean reset

PY ?= python3

help:
	@echo "ClaimGuard"
	@echo ""
	@echo "  make install    install Python dependencies"
	@echo "  make seed       create the database and load sample claims"
	@echo "  make api        run the FastAPI backend on :8000"
	@echo "  make web        run the Next.js dashboard on :3000"
	@echo "  make demo       run the pipeline over seeded claims and print the results"
	@echo ""
	@echo "  make dataset    generate the 150k-claim benchmark dataset"
	@echo "  make train      train the XGBoost fraud model"
	@echo "  make graphsage  train the GraphSAGE claim embeddings"
	@echo "  make evaluate   ablation: base vs graph features vs embeddings"
	@echo ""
	@echo "  make test       run the test suite"
	@echo "  make reset      delete the database and start over"

install:
	$(PY) -m pip install -r requirements.txt

dataset:
	$(PY) ml/generate_dataset.py --claims 150000 --rings 120

seed:
	$(PY) scripts/seed_db.py --claims 400 --reset

api:
	$(PY) -m uvicorn app.main:app --reload --app-dir backend --port 8000

web:
	cd frontend && npm install && npm run dev

demo:
	$(PY) scripts/run_demo.py --limit 25

train:
	$(PY) ml/train_xgboost.py

graphsage:
	$(PY) ml/train_graphsage.py

evaluate:
	$(PY) ml/evaluate.py

test:
	$(PY) -m pytest tests -q

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache frontend/.next

reset:
	rm -f claimguard.db
	rm -rf data/uploads
	@echo "Database removed. Run 'make seed' to start again."
