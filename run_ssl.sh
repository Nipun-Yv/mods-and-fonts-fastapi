#!/bin/bash

# Script to run FastAPI with SSL
# Prefers trusted certificates (mkcert) over self-signed

if [ ! -f "certs/cert.pem" ] || [ ! -f "certs/key.pem" ]; then
    echo "SSL certificates not found."
    echo ""
    echo "For trusted certificates (recommended):"
    echo "  ./setup_trusted_cert.sh"
    echo ""
    echo "For self-signed certificates:"
    echo "  ./generate_ssl_cert.sh"
    exit 1
fi

echo "Starting FastAPI server with SSL..."
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8443 \
    --ssl-keyfile certs/key.pem \
    --ssl-certfile certs/cert.pem \
    --reload
