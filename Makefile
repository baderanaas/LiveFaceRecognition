run:
	python main.py

generate:
	python EncodingGenerator.py

install:
	pip install -r requirements.txt

run-data:
	python data/data_collection.py