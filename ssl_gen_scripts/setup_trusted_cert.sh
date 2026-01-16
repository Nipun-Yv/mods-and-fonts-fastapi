#!/bin/bash

# Script to generate locally-trusted SSL certificate using mkcert
# This creates certificates that browsers will trust automatically

echo "Setting up trusted SSL certificate with mkcert..."

# Check if mkcert is installed
if ! command -v mkcert &> /dev/null; then
    echo "mkcert is not installed."
    echo ""
    echo "Install mkcert:"
    echo "  macOS: brew install mkcert"
    echo "  Linux: See https://github.com/FiloSottile/mkcert#linux"
    echo "  Windows: See https://github.com/FiloSottile/mkcert#windows"
    exit 1
fi

# Create certs directory if it doesn't exist
mkdir -p certs

# Install local CA (if not already installed)
echo "Installing local CA..."
mkcert -install

# Generate certificate for localhost and common variations
echo "Generating trusted certificate for localhost..."
mkcert -key-file certs/key.pem -cert-file certs/cert.pem localhost 127.0.0.1 ::1

echo ""
echo "✅ Trusted SSL certificate generated successfully!"
echo "Certificate: certs/cert.pem"
echo "Private Key: certs/key.pem"
echo ""
echo "These certificates are trusted by your browser automatically!"
echo ""
echo "To run with SSL, use:"
echo "  ./run_ssl.sh"
echo "Or manually:"
echo "  uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem --reload"
