#!/usr/bin/env python3
"""Launch Aurora Trader servers on 0.0.0.0 for LAN access."""
import sys, os, asyncio, signal
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# Load .env before anything else
_env_path = Path(__file__).parent / ".env"
if _env_path.is_file():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# Set host to all interfaces
os.environ["AURORA_HOST"] = "0.0.0.0"

async def learning_main():
    """Start learning server on port 8901."""
    from learning_server.server import LearningServer
    server = LearningServer(host="0.0.0.0", port=8901)
    await server.start()
    while server._running:
        await asyncio.sleep(1)

async def wallet_main():
    """Start wallet scanner on port 8902."""
    from wallet_scanner.scanner import run_scanner
    await run_scanner(host="0.0.0.0", port=8902)

async def main():
    # Cleanup existing processes on our ports
    import psutil
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.laddr.port in [8900, 8901, 8902, 8903]:
                    print(f"🧹 Killing leftover process {proc.pid} on port {conn.laddr.port}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    await asyncio.sleep(1)

    from trading_server.server import main as trading_main
    from integration.server import main as integration_main
    
    # Override the load_config to inject host
    import shared.config as cfg_mod
    orig_load = cfg_mod.load_config
    
    def patched_load(*a, **kw):
        cfg = orig_load(*a, **kw)
        # Patch host values
        if "trading_server" not in cfg.data:
            cfg.data["trading_server"] = {}
        cfg.data["trading_server"]["host"] = "0.0.0.0"
        if "integration" not in cfg.data:
            cfg.data["integration"] = {}
        cfg.data["integration"]["host"] = "0.0.0.0"
        if "learning_server" not in cfg.data:
            cfg.data["learning_server"] = {}
        cfg.data["learning_server"]["host"] = "0.0.0.0"
        return cfg
    
    cfg_mod.load_config = patched_load
    
    print("🚀 Aurora Trader — Starting all servers on 0.0.0.0 (LAN accessible)")
    print("   Learning Server  → http://0.0.0.0:8901")
    print("   Trading Server   → http://0.0.0.0:8900")
    print("   Integration      → http://0.0.0.0:8903")
    print("   Wallet Scanner   → http://0.0.0.0:8902")
    print("   Dashboard        → http://0.0.0.0:8903/dashboard")
    
    await asyncio.gather(
        learning_main(),
        trading_main(),
        integration_main(),
        wallet_main()
    )

if __name__ == "__main__":
    asyncio.run(main())
