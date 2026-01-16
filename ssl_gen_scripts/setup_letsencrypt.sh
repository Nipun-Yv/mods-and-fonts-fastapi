#!/bin/bash

# Script to set up Let's Encrypt certificate for production
# Usage: ./setup_letsencrypt.sh yourdomain.com

if [ -z "$1" ]; then
    echo "Usage: ./setup_letsencrypt.sh yourdomain.com"
    exit 1
fi

DOMAIN=$1

echo "Setting up Let's Encrypt certificate for $DOMAIN..."

# Check if certbot is installed
if ! command -v certbot &> /dev/null; then
    echo "certbot is not installed."
    echo ""
    echo "Install certbot:"
    echo "  Ubuntu/Debian: sudo apt-get install certbot"
    echo "  macOS: brew install certbot"
    echo "  Or visit: https://certbot.eff.org/"
    exit 1
fi

# Stop any service running on port 80 (required for Let's Encrypt)
echo ""
echo "⚠️  Make sure port 80 is available for certificate validation"
echo "   You may need to stop your web server temporarily"
echo ""

# Obtain certificate
echo "Obtaining certificate from Let's Encrypt..."
sudo certbot certonly --standalone -d $DOMAIN

if [ $? -eq 0 ]; then
    CERT_PATH="/etc/letsencrypt/live/$DOMAIN"
    
    echo ""
    echo "✅ Certificate obtained successfully!"
    echo ""
    echo "Certificate location:"
    echo "  Certificate: $CERT_PATH/fullchain.pem"
    echo "  Private Key: $CERT_PATH/privkey.pem"
    echo ""
    echo "To run FastAPI with Let's Encrypt certificate:"
    echo "  sudo uvicorn main:app \\"
    echo "    --host 0.0.0.0 \\"
    echo "    --port 443 \\"
    echo "    --ssl-keyfile $CERT_PATH/privkey.pem \\"
    echo "    --ssl-certfile $CERT_PATH/fullchain.pem"
    echo ""
    echo "To set up auto-renewal, add to crontab:"
    echo "  sudo crontab -e"
    echo "  Add: 0 0 * * * certbot renew --quiet"
else
    echo "❌ Failed to obtain certificate. Check the error messages above."
    exit 1
fi
