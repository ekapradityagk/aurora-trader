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

async def main():
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
        return cfg
    
    cfg_mod.load_config = patched_load
    
    print("🚀 Aurora Trader — Starting both servers on 0.0.0.0 (LAN accessible)")
    print("   Trading Server   → http://0.0.0.0:8900")
    print("   Integration      → http://0.0.0.0:8903")
    print("   Dashboard        → http://0.0.0.0:8903/dashboard")
    
    await asyncio.gather(
        trading_main(),
        integration_main()
    )

if __name__ == "__main__":
    asyncio.run(main())
