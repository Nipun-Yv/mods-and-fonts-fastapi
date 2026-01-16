#!/bin/bash

# Script to generate self-signed SSL certificate for development
# Usage: ./generate_ssl_cert.sh

echo "Generating self-signed SSL certificate for development..."

# Create certs directory if it doesn't exist
mkdir -p certs

# Generate private key
openssl genrsa -out certs/key.pem 2048

# Generate certificate signing request
openssl req -new -key certs/key.pem -out certs/csr.pem -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Generate self-signed certificate (valid for 365 days)
openssl x509 -req -days 365 -in certs/csr.pem -signkey certs/key.pem -out certs/cert.pem

# Clean up CSR file
rm certs/csr.pem

echo "SSL certificate generated successfully!"
echo "Certificate: certs/cert.pem"
echo "Private Key: certs/key.pem"
echo ""
echo "To run with SSL, use:"
echo "uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem"
