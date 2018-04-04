#!/bin/bash
bash <(curl -s https://raw.githubusercontent.com/channelit/launch-scripts/master/aws/docker_ubuntu.sh)

# for re-fetch lfs
# git lfs fetch
docker cp ../python python:/src
docker exec python rm -rf /src
docker exec python nohup python /src/Trajectory.py &
# get counts in each file
# wc -l * | xargs -n2 >> counts.csv
scp -i "cit.pem" -r ubuntu@ec2-34-230-40-91.compute-1.amazonaws.com:/home/ubuntu/elasticsearch-tf-classifier/src/main/resources/large/h_22 .