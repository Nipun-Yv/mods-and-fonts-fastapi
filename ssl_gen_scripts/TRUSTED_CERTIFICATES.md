# Trusted SSL Certificates Setup

This guide shows you how to get **real, trusted SSL certificates** instead of self-signed ones.

## Option 1: mkcert (Recommended for Development)

**mkcert** creates locally-trusted certificates that your browser will accept automatically - no warnings!

### Installation

**macOS:**
```bash
brew install mkcert
```

**Linux:**
```bash
# Follow instructions at: https://github.com/FiloSottile/mkcert#linux
# Or use snap:
sudo snap install mkcert
```

**Windows:**
```bash
# Using Chocolatey
choco install mkcert

# Or download from: https://github.com/FiloSottile/mkcert/releases
```

### Setup

1. **Run the setup script:**
   ```bash
   ./setup_trusted_cert.sh
   ```

   This will:
   - Install mkcert's local CA (one-time setup)
   - Generate trusted certificates for localhost
   - Place them in the `certs/` directory

2. **Run your server:**
   ```bash
   ./run_ssl.sh
   ```

3. **Access your API:**
   - Visit `https://localhost:8443` in your browser
   - **No certificate warnings!** ✅

### Manual Setup

If you prefer to do it manually:

```bash
# Install local CA (one-time)
mkcert -install

# Generate certificates
mkdir -p certs
mkcert -key-file certs/key.pem -cert-file certs/cert.pem localhost 127.0.0.1 ::1
```

## Option 2: Let's Encrypt (For Production)

**Let's Encrypt** provides free, publicly-trusted certificates for production use.

### Prerequisites

- A domain name pointing to your server
- Port 80 accessible from the internet
- Server with public IP

### Setup

1. **Install certbot:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install certbot

   # macOS
   brew install certbot
   ```

2. **Obtain certificate:**
   ```bash
   ./setup_letsencrypt.sh yourdomain.com
   ```

   Or manually:
   ```bash
   sudo certbot certonly --standalone -d yourdomain.com
   ```

3. **Certificates location:**
   ```
   /etc/letsencrypt/live/yourdomain.com/fullchain.pem
   /etc/letsencrypt/live/yourdomain.com/privkey.pem
   ```

4. **Run FastAPI with Let's Encrypt:**
   ```bash
   sudo uvicorn main:app \
       --host 0.0.0.0 \
       --port 443 \
       --ssl-keyfile /etc/letsencrypt/live/yourdomain.com/privkey.pem \
       --ssl-certfile /etc/letsencrypt/live/yourdomain.com/fullchain.pem
   ```

### Auto-Renewal

Let's Encrypt certificates expire after 90 days. Set up auto-renewal:

```bash
# Test renewal
sudo certbot renew --dry-run

# Add to crontab
sudo crontab -e
# Add this line:
0 0 * * * certbot renew --quiet && systemctl reload nginx
```

## Comparison

| Method | Trust Level | Use Case | Browser Warnings |
|--------|------------|----------|------------------|
| **Self-signed** | None | Testing only | ⚠️ Always shows |
| **mkcert** | Local trust | Development | ✅ None |
| **Let's Encrypt** | Public trust | Production | ✅ None |

## Troubleshooting

### mkcert Issues

**"mkcert: command not found"**
- Install mkcert first (see installation above)

**Certificate still shows warning**
- Make sure you ran `mkcert -install` first
- Restart your browser
- Clear browser cache

### Let's Encrypt Issues

**"Port 80 already in use"**
- Stop your web server temporarily
- Or use webroot method instead of standalone

**"Domain validation failed"**
- Ensure DNS points to your server
- Ensure port 80 is accessible from internet
- Check firewall settings

## Security Notes

- **mkcert certificates** are only trusted on your local machine
- **Let's Encrypt certificates** are trusted by all browsers worldwide
- Never commit private keys to git (already in `.gitignore`)
- Keep certificates updated and renewed

## Next Steps

1. **Development:** Use `./setup_trusted_cert.sh` with mkcert
2. **Production:** Use `./setup_letsencrypt.sh` with Let's Encrypt
3. **Run server:** Use `./run_ssl.sh` (works with both)
