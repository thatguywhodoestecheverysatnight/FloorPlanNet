.PHONY: install dev test lint data synth train-demo train eval export serve docker web eval-vector standalone
PY ?= python

install:          ## training + inference deps
	pip install -r requirements.txt && pip install -e .
dev:
	pip install -r requirements-dev.txt && pip install -e .
test:
	pytest
lint:
	ruff check floorplannet serve tests scripts
data:             ## download + rasterise CubiCasa5K
	bash scripts/download_cubicasa5k.sh data
synth:            ## render the synthetic training set
	$(PY) scripts/make_synthetic.py --out data/synthetic --train 3000 --val 200 --workers 8
train-demo: synth ## browser model (CPU friendly)
	floorplannet train -c configs/synthetic_lite.yaml
	$(PY) scripts/export_web_model.py --ckpt runs/synthetic_lite/best.pt
train:            ## main benchmark model
	floorplannet train -c configs/cubicasa5k_deeplabv3_r101.yaml
eval:
	floorplannet evaluate -c configs/cubicasa5k_deeplabv3_r101.yaml --ckpt runs/cc5k_deeplabv3_r101/best.pt --split test --tta
export:
	floorplannet export --ckpt runs/cc5k_deeplabv3_r101/best.pt --out models/floorplannet_r101.onnx
serve:
	uvicorn serve.app:app --reload --port 8000
docker:
	docker compose up --build
web:
	cd web && npm install && npm run dev
eval-vector:      ## instance-level room / door / window F1 of the full pipeline
	$(PY) scripts/eval_vectorization.py --model models/floorplannet_lite.onnx --data data/synthetic --n 200
standalone:       ## static build of the web app with relative paths (any static host)
	cd web && npm run build:standalone
