COMPOSE := docker compose -f configs/relcompat3d/compose.yaml

.PHONY: build validate models open3dsg-checkpoint reproduce export-rows train evaluate

build:
	$(COMPOSE) build relcompat3d_reproduce_rows

validate:
	scripts/validate.sh

models:
	scripts/download_models.sh

open3dsg-checkpoint:
	scripts/download_open3dsg_checkpoint.sh

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
