#!/usr/bin/env bash

docker build --no-cache --network host -t warhammerizator:dev -f Dockerfile .
