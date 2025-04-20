#!/bin/bash

# Stop any existing containers
docker-compose down

# Start the container in detached mode
docker-compose up -d --build

# Wait for container to start
sleep 5

# Get the allocated port
PORT=$(docker port scout_model_scout_model | grep '8000/tcp' | cut -d ':' -f2)

if [ -z "$PORT" ]; then
    echo "Error: Could not find port mapping for container scout_model_scout_model"
    exit 1
fi

echo "Container scout_model is running on port $PORT"

# Follow logs
docker-compose logs -f
