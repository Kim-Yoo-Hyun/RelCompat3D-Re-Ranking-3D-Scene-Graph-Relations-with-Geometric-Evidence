COMPOSE := docker compose -f configs/relcompat3d/compose.yaml

.PHONY: build validate models open3dsg-train reproduce export-rows train evaluate

build:
	$(COMPOSE) build relcompat3d_reproduce_rows

validate:
	scripts/validate.sh

models:
	scripts/download_models.sh

open3dsg-train:
	scripts/train_open3dsg.sh all

reproduce:
	scripts/reproduce_tables.sh

export-rows:
	$(COMPOSE) run --rm relcompat3d_export_rows

train:
	$(COMPOSE) run --rm relcompat3d_fit
	$(COMPOSE) run --rm relcompat3d_freeze_initial
	scripts/run_pipeline.sh initial

evaluate:
	scripts/run_pipeline.sh downstream
