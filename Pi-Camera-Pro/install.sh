#!/bin/bash

echo "═══════════════════════════════════════════════════════════"
echo "  🚀 ULTRA Pi Camera Pro - Installation Script"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    echo "   Camera functionality may not work properly"
    echo ""
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Check installation
echo ""
echo "✅ Installation complete!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  🎯 Next Steps:"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1. Run the server:"
echo "   python3 camserver.py"
echo ""
echo "2. Open in browser:"
echo "   http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "3. Login credentials:"
echo "   Username: admin"
echo "   Password: picam2025"
echo ""
echo "4. IMPORTANT: Change the password in camserver.py!"
echo ""
echo "═══════════════════════════════════════════════════════════"
