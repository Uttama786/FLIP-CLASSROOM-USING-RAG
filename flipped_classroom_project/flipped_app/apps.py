from django.apps import AppConfig


class FlippedAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'flipped_app'

    def ready(self):
        """
        Called once Django is fully loaded.
        Register real-time dataset signals and configure SQLite for concurrency.
        """

        # ── 1. SQLite WAL mode (prevents "database is locked" errors) ────────
        # WAL allows concurrent reads while background ML threads write,
        # eliminating the OperationalError: database is locked crashes.
        try:
            from django.db.backends.signals import connection_created

            def _set_wal(sender, connection, **kwargs):
                if connection.vendor == 'sqlite':
                    try:
                        cursor = connection.cursor()
                        cursor.execute('PRAGMA journal_mode=WAL;')
                        cursor.execute('PRAGMA synchronous=NORMAL;')
                        cursor.execute('PRAGMA busy_timeout=30000;')  # 30 s
                    except Exception:
                        pass

            connection_created.connect(_set_wal)
        except Exception as e:
            print(f"[FlipLearn] WAL mode setup skipped: {e}")

        # ── 2. Register signals ──────────────────────────────────────────────
        try:
            import flipped_app.signals  # noqa: F401  (side-effect import)
            print("[FlipLearn] Real-time dataset signals registered ✓")
        except Exception as e:
            print(f"[FlipLearn] Signal registration skipped: {e}")

        # ── 3. Pre-warm RAG retriever ────────────────────────────────────────
        # Disabled: model download (~90MB) at startup causes health check
        # timeouts on Railway. The retriever loads lazily on first chat request.
        pass
