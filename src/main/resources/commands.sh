#!/usr/bin/env bash
docker run -it -p 8888:8888 -p 6006:6006 --name=tensorflow -v $(pwd)/data:/data -e PASSWORD=password cithub/tensorflow

# ELASTIC DOCKER CLUSTER COMMANDLINE
# docker run -d --network=host -e TAKE_FILE_OWNERSHIP=true -e cluster.name=xxx -e node.name=xxx -e network.host=0.0.0.0 -e discovery.zen.ping.unicast.hosts="0.0.0.0","x.x.x.x" -e transport.tcp.port=990x -e http.port=990x -p 990x:990x -p 990x:990x -v /full/data:/usr/share/elasticsearch/data -v /full/path:/usr/share/elasticsearch/logs --name=elastic elasticsearch/image