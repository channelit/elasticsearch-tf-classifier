#!/bin/bash
bash <(curl -s https://raw.githubusercontent.com/channelit/launch-scripts/master/aws/docker_ubuntu.sh)
docker cp ../python python:/src
docker exec python python /src/Trajectory.py