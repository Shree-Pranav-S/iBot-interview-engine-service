# Makefile for interview-engine-service

.PHONY: run
run:
	uvicorn src.api.rest.app:app --host 0.0.0.0 --port 8001 --reload
