"""Точка входа: python -m forky_spoonky_pi"""

from __future__ import annotations

import sys

from .config import load_config
from .logging_setup import get_logger, setup_logging
from .web import create_app


def main() -> int:
    config = load_config()
    paths = config.resolved_paths.ensure()
    setup_logging(
        level=config.logging.level,
        log_file=paths.logs / "app.log",
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
    )
    log = get_logger("forky_spoonky_pi")
    log.info(
        "Старт: http://%s:%s (data=%s, model=%s)",
        config.server.host,
        config.server.port,
        paths.data,
        config.model.weights,
    )

    app = create_app(config)
    try:
        app.run(host=config.server.host, port=config.server.port, threaded=True)
    except KeyboardInterrupt:
        log.info("Остановлено пользователем")
    return 0


if __name__ == "__main__":
    sys.exit(main())
