#!/bin/bash

set -e

cd /opt/myapp

docker-compose -f docker-compose.production.yml pull

docker-compose -f docker-compose.production.yml up -d

docker image prune -f
