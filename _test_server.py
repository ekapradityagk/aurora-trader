"""Quick script to start the integration server for testing."""
import asyncio
from integration.server import IntegrationServer


async def main():
    server = IntegrationServer()
    await server.start()
    print("SERVER_READY", flush=True)
    # Keep running
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
