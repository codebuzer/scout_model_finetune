#!/bin/bash

# Create scripts directory and script
mkdir -p scripts
cat > scripts/get_container_port.sh << 'EOF'
#!/bin/bash

CONTAINER_NAME="scout_model_scout_model"

# Get the dynamic port mapping
PORT=$(docker port $CONTAINER_NAME | grep '8000/tcp' | cut -d ':' -f2)

if [ -z "$PORT" ]; then
    echo "Error: Could not find port mapping for container $CONTAINER_NAME"
    exit 1
fi

echo "Container $CONTAINER_NAME is running on port $PORT"
EOF

# Make the script executable
chmod +x scripts/get_container_port.sh
