COMPOSE := docker compose -f configs/relcompat3d/compose.yaml

.PHONY: build validate models prepare open3dsg-train reproduce export-rows train evaluate evaluate-trained audit-trained tables-trained

build:
	$(COMPOSE) build relcompat3d_reproduce_rows

validate:
	scripts/validate.sh

models:
	scripts/download_models.sh

prepare:
	scripts/run_pipeline.sh prepare

open3dsg-train:
	scripts/train_open3dsg.sh all

reproduce:
	scripts/reproduce_tables.sh

export-rows:
	$(COMPOSE) run --rm relcompat3d_export_rows

train:
	scripts/run_pipeline.sh train

evaluate:
	scripts/run_pipeline.sh evaluate

evaluate-trained:
	scripts/run_pipeline.sh evaluate-trained

audit-trained:
	scripts/run_pipeline.sh audit-trained

tables-trained:
	scripts/run_pipeline.sh tables-trained
