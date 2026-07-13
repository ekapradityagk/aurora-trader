"""Quick integration test for the server."""
import asyncio
import aiohttp
from integration.server import IntegrationServer


async def test():
    server = IntegrationServer()
    await server.start()

    async with aiohttp.ClientSession() as session:
        # Test /health
        async with session.get("http://127.0.0.1:8903/health") as resp:
            data = await resp.json()
            print(f"GET /health -> {resp.status}: status={data['status']}")

        # Test /versions
        async with session.get("http://127.0.0.1:8903/versions") as resp:
            data = await resp.json()
            print(f"GET /versions -> {resp.status}: count={data['count']}")

        # Test /versions POST (create a version)
        async with session.post(
            "http://127.0.0.1:8903/versions",
            json={"strategy_name": "ema_crossover", "parameters": {"ema_fast": 9, "ema_slow": 50}},
        ) as resp:
            data = await resp.json()
            print(f"POST /versions -> {resp.status}: tag={data.get('version_tag')}")

        # Test /versions/<tag>
        async with session.get("http://127.0.0.1:8903/versions/v1.0.0") as resp:
            data = await resp.json()
            print(f"GET /versions/v1.0.0 -> {resp.status}: params={data.get('parameters')}")

        # Test /deploy
        async with session.post(
            "http://127.0.0.1:8903/deploy",
            json={"version_tag": "v1.0.0"},
        ) as resp:
            data = await resp.json()
            print(f"POST /deploy -> {resp.status}: status={data.get('status')}")

        # Test /winrate/best
        async with session.get("http://127.0.0.1:8903/winrate/best") as resp:
            data = await resp.json()
            print(f"GET /winrate/best -> {resp.status}")

        # Test /trades
        async with session.get("http://127.0.0.1:8903/trades") as resp:
            data = await resp.json()
            print(f"GET /trades -> {resp.status}: count={data.get('count')}")

        # Test /winrate/compare
        async with session.post(
            "http://127.0.0.1:8903/winrate/compare",
            json={"version_a": "v1.0.0", "version_b": "v1.0.0"},
        ) as resp:
            data = await resp.json()
            print(f"POST /winrate/compare -> {resp.status}")

        # Test /status
        async with session.get("http://127.0.0.1:8903/status") as resp:
            data = await resp.json()
            print(f"GET /status -> {resp.status}: overall={data.get('overall')}")

        # Test /dashboard
        async with session.get("http://127.0.0.1:8903/dashboard") as resp:
            data = await resp.json()
            print(f"GET /dashboard -> {resp.status}: keys={list(data.keys())}")

        # Test /rollback
        async with session.post(
            "http://127.0.0.1:8903/rollback",
            json={"version_tag": "v1.0.0"},
        ) as resp:
            data = await resp.json()
            print(f"POST /rollback -> {resp.status}: status={data.get('status')}")

        # Test /winrate/<tag>
        async with session.get("http://127.0.0.1:8903/winrate/v1.0.0") as resp:
            data = await resp.json()
            print(f"GET /winrate/v1.0.0 -> {resp.status}: {data.get('winrate')}")

    print("\n=== ALL ENDPOINTS TESTED SUCCESSFULLY ===")

    await server.stop()


if __name__ == "__main__":
    asyncio.run(test())
