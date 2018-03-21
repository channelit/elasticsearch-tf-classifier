#!/bin/bash
bash <(curl -s https://raw.githubusercontent.com/channelit/launch-scripts/master/aws/docker_ubuntu.sh)

# for re-fetch lfs
# git lfs fetch
docker cp ../python python:/src
docker exec python python /src/Trajectory.py -d
scp -i "cit.pem" -r ubuntu@ec2-54-175-79-187.compute-1.amazonaws.com:/home/ubuntu/elasticsearch-tf-classifier/src/main/resources/data/tensorflow .