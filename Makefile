COMPOSE := docker compose -f configs/relcompat3d/compose.yaml

.PHONY: build validate prepare open3dsg-train train evaluate audit tables full

build:
	$(COMPOSE) build relcompat3d_generate_tables

validate:
	scripts/validate.sh

prepare:
	scripts/run_pipeline.sh prepare

open3dsg-train:
	scripts/train_open3dsg.sh all

train:
	scripts/run_pipeline.sh train

evaluate:
	scripts/run_pipeline.sh evaluate

audit:
	scripts/run_pipeline.sh audit

tables:
	scripts/run_pipeline.sh tables

full:
	scripts/run_pipeline.sh full
