PYTHON_VERSION := $(shell cat .tool-versions| grep -i 'python' | cut -d' ' -f2)
PYTHON_MAJOR_VERSION := $(shell cat .tool-versions| grep -i 'python' | cut -d' ' -f2 | cut -d'.' -f1,2)
POETRY_VERSION := $(shell cat .tool-versions| grep -i 'poetry' | cut -d' ' -f2)

# Usage: make setup
# Description: Installs the specified versions of Python and Poetry using asdf, and installs project dependencies with Poetry.
# Requires: .tool-versions file with specified Python and Poetry versions.
setup:
	asdf install python ${PYTHON_VERSION}
	asdf local python ${PYTHON_VERSION}
	pip install poetry==${POETRY_VERSION}
	poetry install

# Usage: make lint
# Description: Runs lint checks
# Requires: .tool-versions file with specified Python and Poetry versions.
lint:
	poetry run ruff check .
	poetry run mypy .
	poetry run coverage run -m --source ml_template pytest 
	poetry run coverage report -m
	
# Usage: make format
# Description: Fix imports and indentation
# Requires: .tool-versions file with specified Python and Poetry versions.
format:
	poetry run isort ml_template tests server
	poetry run ruff format ml_template tests server

# Usage: make test
# Description: Run pytests
# Requires: .tool-versions file with specified Python and Poetry versions.
test:
	poetry run coverage run -m pytest 

# Usage: make build_docker
# Description: build docker container with ml_template codebase. The docker container can be used either for 
# training or for deploying model as a service
# Requires: .tool-versions file with specified Python and Poetry versions.
build_docker:
	docker build -f docker/base.docker --build-arg PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} --build-arg POETRY_PACKAGE_VERSION=${POETRY_VERSION} -t ml_template .

# Usage: make train
# Description: Trains ML models for a given language. 
# Usage
# make language=english train
# make language=brazil train
# Requires: 
# 1. .tool-versions file with specified Python and Poetry versions. 
# 2. Download necessary data files in data folder.
train: build_docker

# NOTE -- In makefile do not indent if then else
ifeq ( $(language), "english")
train_file="data/english/flexWords.tsv"
else ifeq ( $(language), "brazil")
train_file="data/brazil/train.feather"
endif


	docker run \
	-v ${PWD}/data:/app/data \
	-v ${PWD}/output:/app/output \
	ml_template \
	poetry run python -m ml_template.cli.train \
	${language} \
	--train_file ${train_file} \
	--output_dir output/${language}/


# Usage: make run_server
# Description: Deploy model as a service
# Requires: 
# 1. .tool-versions file with specified Python and Poetry versions. 
# 2. Trained model should be available in the output folder. 
# Usage:
# make model_file=output/brazil/brazil_Risk.pkl run_server
#
run_server: build_docker
	docker run \
		-v ${PWD}/output:/app/output \
		-p 80:80 \
		--env MODEL=${model_file} \
		ml_template \
		poetry run uvicorn server.main:app --host 0.0.0.0 --port 80



