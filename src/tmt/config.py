"""Default model settings for CLI configuration and checkpoint validation."""

MODEL_DEFAULTS = {
    'dim': 512,
    'layers': 16,
    'spread': 32,
    'temp': 0.75,
    'rate': 0.0005,
    'bound': (40000, 120000),
    'block': 'rtu',
}
