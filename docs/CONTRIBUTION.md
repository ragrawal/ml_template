# Developer's Guide

## Contribution guide
1. Make sure that all code format and lints checks (`make lints`) are passing
2. Make sure to update unit tests and all tests are passing (`make tests`)



## Setting up
This repo uses 
* [ASDF](https://asdf-vm.com/) for python version management
* [poetry](https://python-poetry.org/) for dependency resolution and management
* pytest for unit testing. 

Follow these steps to setup the repo:
```
# 1. Install ASDF. For mac, use brew 
# TODO: haven't tested if the below command creates right environment setup
brew install asdf 

# 2. Add Python Plugin
asdf plugin-add python

# 3. Setup development environment using makefile 
make setup

```

## Training your first model
You can use `make train` command to train your first model. Follow the `train` command in the Makefile. In your for
`make train` to successfully complete, you will need training file placed at "data/english/flexWords.tsv". Again please refer `train` command in the Makefile to fully understand how the training routine runs.

## Deploying the model
Assuming you have successfully ran `make train`, now you can run `make run_server`. This will deploy the model as a service using a docker container. From command line, run the following curl command to test the service.

```
curl -X 'POST' \
  'http://localhost/english/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[{"post":"hello|world"}]' 
```

Also checkout other [API](http://localhost/docs)


## FAQ

**How to add a new python dependency?**
if the package is only needed for development/debug or testing, add it to the dev group using the below command
```poetry add <package-name> --group dev```

if the package is needed for the core functionality, use the below command
```poetry add <package-name> [version]```



