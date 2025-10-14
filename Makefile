# CryoMamba Development Makefile

.PHONY: help build up down logs test clean dev-install lint format

# Default target
help:
	@echo "CryoMamba Development Commands:"
	@echo "  make build        - Build Docker image"
	@echo "  make up           - Start development server"
	@echo "  make down         - Stop development server"
	@echo "  make logs         - View server logs"
	@echo "  make test         - Run health check tests"
	@echo "  make clean        - Clean Docker resources"
	@echo "  make dev-install  - Install local dependencies"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code"

# Docker commands
build:
	docker-compose build

up:
	docker-compose up --build

down:
	docker-compose down

logs:
	docker-compose logs -f cryomamba-server

# Development commands
dev-install:
	pip install -r requirements.txt

# Testing commands
test:
	@echo "Testing server health..."
	@curl -f http://localhost:8000/v1/healthz || echo "Health check failed"
	@echo "Testing server info..."
	@curl -f http://localhost:8000/v1/server/info || echo "Server info failed"

# Code quality commands
lint:
	@echo "Running linting..."
	@python -m flake8 app/ --max-line-length=88 --extend-ignore=E203,W503
	@python -m black --check app/

format:
	@echo "Formatting code..."
	@python -m black app/

# Cleanup commands
clean:
	docker-compose down -v
	docker system prune -f

# Development workflow
dev: build up

restart: down up

# Quick health check
health:
	@curl -s http://localhost:8000/v1/healthz | python -m json.tool || echo "Server not responding"
