COMPOSE := docker compose -f configs/relcompat3d/compose.yaml

.PHONY: build validate models reproduce export-rows train evaluate

build:
	$(COMPOSE) build relcompat3d_reproduce_rows

validate:
	scripts/validate.sh

models:
	scripts/download_models.sh

reproduce:
	scripts/reproduce_tables.sh

export-rows:
	$(COMPOSE) run --rm relcompat3d_export_rows

train:
	$(COMPOSE) run --rm no_family_indicator_fit
	$(COMPOSE) run --rm no_family_indicator_freeze_initial
	scripts/run_pipeline.sh initial

evaluate:
	scripts/run_pipeline.sh downstream
