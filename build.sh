#!/usr/bin/env bash

docker build --no-cache --debug --network host -t warhammerizator:dev -f Dockerfile .
