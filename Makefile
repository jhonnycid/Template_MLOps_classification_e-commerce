# === Entraînement local via Dockerfile.dev ===
train:
	@echo "🔍 Recherche du réseau Docker pour MLflow..."
	NET_NAME=$$(docker network ls --format '{{.Name}}' | grep ml-network || echo "bridge"); \
	echo "🌐 Réseau détecté: $$NET_NAME"; \
	docker build -t rakuten-dev -f Dockerfile.dev . && \
	docker run --rm \
		--network $$NET_NAME \
		-v "$(PWD)":/app \
		-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
		rakuten-dev

# === Prédiction (via predict.py) ===
predict:
	@echo "📦 Lancement des prédictions..."
	docker exec -i $$(docker ps -qf "name=api-fast") python src/predict.py

# === Monitoring via conteneur dédié ===
monitor:
	@echo "📊 Monitoring des dérives avec Evidently..."
	docker-compose run --rm monitor

# === API FastAPI ===
api:
	docker build -t rakuten-api -f Dockerfile.api .
	docker run --rm -p 8000:8000 -v "$(PWD)":/app \
		-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
		rakuten-api

check-api:
	curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"input": "sample text"}'

# === Dashboard CLI interactif (auto) ===
dashboard:
	@echo "🚀 Lancement du Dashboard CLI auto..."
	python dashboard.py

# === Lancer tous les services en continu ===
up:
	docker-compose up --build -d
	@make dashboard

down:
	docker-compose down -v

# === Pipeline complet manuel ===
full-run:
	@make train
	@make predict
	@make monitor
	@make dashboard

# === Pipeline complet + API ===
full-run-api:
	@make train
	@make predict
	@make monitor
	@make api
	@make dashboard

# === Pipeline complet sans entraînement (modèles déjà présents) ===
no-train-run:
	@make predict
	@make monitor
	@make dashboard

no-train-run-api:
	@make predict
	@make monitor
	@make api
	@make dashboard

# === Pipeline full-stack Docker Compose ===
orchestrated:
	docker-compose down -v
	docker-compose up --build

# === Qualité de code ===
lint:
	black . --check
	flake8 .

format:
	black .

test:
	pytest tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf models/__pycache__/

nltk-init:
	docker run --rm -v "$(PWD)":/app rakuten-dev python -c "import nltk; nltk.download('punkt')"
