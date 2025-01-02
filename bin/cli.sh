#!/bin/bash
docker-compose run --rm python-dev python3 src/house_bot/enrichment/cli.py "$@"