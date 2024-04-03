run:
	python main.py

generate:
	python EncodingGenerator.py

install:
	pip install -r requirements.txt

install-data:
	pip install -r data/requirements.txt

run-data:
	python data/data_collection.py