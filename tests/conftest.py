"""Keep the suite independent of a developer's local .env.

pydantic-settings reads `.env` on every `Settings()` call, so a filled-in local
file would switch the provider on mid-test and send real requests. Tests set
exactly the environment they need with monkeypatch instead.
"""

from aevum_ai.config import Settings

Settings.model_config["env_file"] = None
