install:
	pip install -r requirements.txt

test:
	pytest tests/

clean:
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"

run-lab1:
	python labs/theme_01_preprocessing/run_pipeline.py