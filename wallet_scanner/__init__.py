"""
Aurora Trader — Wallet Scanner.

Monitors on-chain data, CEX exchange flows, whale wallet activity, and
perpetual futures funding rates to produce a combined bullish/bearish bias
score for configured trading symbols.

Sub-modules:
    scanner             Main async HTTP server (port 8902)
    exchange_flow       Tracks CEX exchange inflows/outflows
    whale_tracker       Monitors whale wallet accumulation/distribution
    funding_rate        Fetches perpetual futures funding rates
    signal_aggregator   Combines all signals into a -10 to +10 score
"""

from wallet_scanner.scanner import WalletScanner, run_scanner

__all__ = [
    "WalletScanner",
    "run_scanner",
]
