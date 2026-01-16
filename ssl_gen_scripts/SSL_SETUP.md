# SSL Setup Guide for FastAPI Backend

This guide covers setting up SSL/TLS for your FastAPI backend in both development and production environments.

## Development Setup (Self-Signed Certificate)

### Option 1: Using the provided script

1. **Make the script executable:**
   ```bash
   chmod +x generate_ssl_cert.sh
   chmod +x run_ssl.sh
   ```

2. **Generate SSL certificates:**
   ```bash
   ./generate_ssl_cert.sh
   ```

3. **Run the server with SSL:**
   ```bash
   ./run_ssl.sh
   ```

   Or manually:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem --reload
   ```

### Option 2: Manual certificate generation

```bash
# Create certs directory
mkdir -p certs

# Generate private key
openssl genrsa -out certs/key.pem 2048

# Generate certificate
openssl req -new -x509 -key certs/key.pem -out certs/cert.pem -days 365 -subj "/CN=localhost"
```

### Accessing the API

- **HTTPS URL:** `https://localhost:8443`
- **Note:** Browsers will show a security warning for self-signed certificates. Click "Advanced" → "Proceed to localhost" to continue.

## Production Setup

### Option 1: Using Let's Encrypt (Recommended)

1. **Install Certbot:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install certbot

   # macOS
   brew install certbot
   ```

2. **Obtain certificate:**
   ```bash
   sudo certbot certonly --standalone -d yourdomain.com
   ```

3. **Certificates will be stored at:**
   - Certificate: `/etc/letsencrypt/live/yourdomain.com/fullchain.pem`
   - Private Key: `/etc/letsencrypt/live/yourdomain.com/privkey.pem`

4. **Run uvicorn with Let's Encrypt certificates:**
   ```bash
   sudo uvicorn main:app \
       --host 0.0.0.0 \
       --port 443 \
       --ssl-keyfile /etc/letsencrypt/live/yourdomain.com/privkey.pem \
       --ssl-certfile /etc/letsencrypt/live/yourdomain.com/fullchain.pem
   ```

5. **Auto-renewal setup:**
   ```bash
   # Test renewal
   sudo certbot renew --dry-run

   # Add to crontab for auto-renewal
   sudo crontab -e
   # Add: 0 0 * * * certbot renew --quiet
   ```

### Option 2: Using Nginx as Reverse Proxy (Recommended for Production)

This is often better than running uvicorn directly with SSL because:
- Nginx handles SSL termination
- Better performance for static files
- Easier certificate management
- Can run multiple services behind one SSL endpoint

1. **Install Nginx:**
   ```bash
   sudo apt-get install nginx
   ```

2. **Create Nginx configuration** (`/etc/nginx/sites-available/fastapi`):
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       
       # Redirect HTTP to HTTPS
       return 301 https://$server_name$request_uri;
   }

   server {
       listen 443 ssl http2;
       server_name yourdomain.com;

       ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
       
       # SSL configuration
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers HIGH:!aNULL:!MD5;
       ssl_prefer_server_ciphers on;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

3. **Enable the site:**
   ```bash
   sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

4. **Run FastAPI without SSL** (Nginx handles it):
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```

## Environment Variables

You can also configure SSL via environment variables:

```bash
export SSL_KEYFILE="certs/key.pem"
export SSL_CERTFILE="certs/cert.pem"
export SSL_PORT=8443
```

Then modify your startup script to use these variables.

## Testing SSL

```bash
# Test with curl
curl -k https://localhost:8443/health

# Test with openssl
openssl s_client -connect localhost:8443 -showcerts
```

## Troubleshooting

1. **Permission errors:**
   - Make sure certificate files are readable
   - For Let's Encrypt, may need `sudo` or proper permissions

2. **Port already in use:**
   - Change port in uvicorn command
   - Or stop the service using port 443/8443

3. **Certificate errors:**
   - Ensure certificate and key paths are correct
   - Check certificate expiration date
   - Verify certificate matches the domain

4. **Browser warnings (self-signed):**
   - This is normal for development
   - In production, use Let's Encrypt for trusted certificates

## Security Notes

- **Never commit private keys to git**
- Add `certs/` to `.gitignore`
- Use strong passwords for production keys
- Keep certificates updated
- Use TLS 1.2+ in production
- Consider using HSTS headers
