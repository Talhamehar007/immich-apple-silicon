"""Both servers are unauthenticated; binding them is the only control.

So the config key that chooses an address has to reach the socket, and nothing
else may be silently substituted for it. These also pin the defaults: changing
them would either expose a box that was private or break the documented split
deployment, where Immich in Docker reaches this Mac through
host.docker.internal rather than loopback.
"""

from __future__ import annotations

import argparse
import sys
import types
from unittest.mock import patch

import pytest

import immich_accelerator.__main__ as m
from immich_accelerator.common import (
    BIND_ANY,
    BIND_LOOPBACK,
    bind_host,
    insecure_immich_url,
    is_loopback_url,
)


class TestBindHost:
    def test_absent_keeps_the_documented_default(self):
        assert bind_host({}, "dashboard_host") == BIND_ANY
        assert bind_host({}, "ml_host") == BIND_ANY

    def test_loopback_is_honoured(self):
        assert bind_host({"ml_host": "127.0.0.1"}, "ml_host") == BIND_LOOPBACK

    def test_wildcard_is_honoured(self):
        assert bind_host({"ml_host": "0.0.0.0"}, "ml_host") == BIND_ANY

    def test_surrounding_space_is_trimmed(self):
        assert bind_host({"ml_host": " 127.0.0.1\n"}, "ml_host") == BIND_LOOPBACK

    @pytest.mark.parametrize(
        "value", ["192.168.1.5", "localhost", "::1", "", 5, True, ["127.0.0.1"]]
    )
    def test_anything_else_falls_back_instead_of_reaching_the_socket(self, value):
        # Every liveness check in __main__ probes 127.0.0.1 and ::1. A listener
        # bound to one LAN address answers neither, so the watcher would call a
        # healthy service down and start a second copy beside it.
        assert bind_host({"ml_host": value}, "ml_host") == BIND_ANY


class TestInsecureImmichUrl:
    def test_plain_http_to_another_machine_is_reported(self):
        assert insecure_immich_url({"immich_url": "http://nas:2283"}) == "http://nas:2283"

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:2283",
            "http://127.0.0.1:2283",
            "http://[::1]:2283",
            "https://nas:2283",
            "https://photos.example.com",
        ],
    )
    def test_nothing_to_say(self, url):
        assert insecure_immich_url({"immich_url": url}) is None

    def test_absent_or_blank(self):
        assert insecure_immich_url({}) is None
        assert insecure_immich_url({"immich_url": "   "}) is None

    def test_loopback_names(self):
        assert is_loopback_url("http://localhost:2283")
        assert is_loopback_url("http://127.0.0.1:2283")
        assert not is_loopback_url("http://nas:2283")


class TestDashboardBind:
    def test_default_is_every_interface(self, saved_config):
        assert m._dashboard_host() == BIND_ANY

    def test_config_loopback_is_used(self, saved_config):
        saved_config["dashboard_host"] = BIND_LOOPBACK
        m.save_config(saved_config)
        assert m._dashboard_host() == BIND_LOOPBACK

    def test_start_dashboard_passes_the_host(self, saved_config, tmp_data_dir):
        saved_config["dashboard_host"] = BIND_LOOPBACK
        m.save_config(saved_config)
        with patch.object(m, "read_pid", return_value=None), patch.object(
            m, "_pid_on_port", return_value=None
        ), patch("subprocess.Popen") as popen, patch.object(m, "write_pid"):
            assert m.start_dashboard() is True
        argv = popen.call_args.args[0]
        assert argv[argv.index("--host") + 1] == BIND_LOOPBACK

    def test_run_dashboard_passes_the_host_to_uvicorn(self, monkeypatch):
        from immich_accelerator import dashboard

        seen = {}
        fake = types.SimpleNamespace(
            run=lambda app, host, port, log_level: seen.update(host=host, port=port)
        )
        monkeypatch.setitem(sys.modules, "uvicorn", fake)
        dashboard.run_dashboard({"version": "test"}, port=8420, host=BIND_LOOPBACK)
        assert seen == {"host": BIND_LOOPBACK, "port": 8420}

    def test_cmd_dashboard_honours_an_explicit_host(self, saved_config, monkeypatch):
        from immich_accelerator import dashboard

        seen = {}
        monkeypatch.setattr(
            dashboard, "run_dashboard",
            lambda config, port, host: seen.update(port=port, host=host),
        )
        m.cmd_dashboard(argparse.Namespace(state=None, port=9000, host=BIND_LOOPBACK))
        assert seen == {"port": 9000, "host": BIND_LOOPBACK}


class TestMLBind:
    def _fake_native(self, monkeypatch, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        clip = tmp_path / "clip"
        clip.mkdir()
        (clip / "model.safetensors").write_bytes(b"x")
        arcface = tmp_path / "arcface.onnx"
        arcface.write_bytes(b"x")
        monkeypatch.setattr(m, "_native_bundle_dir", lambda config: bundle)
        monkeypatch.setattr(m, "_native_clip_dir", lambda config: clip)
        monkeypatch.setattr(m, "_native_arcface", lambda config: arcface)

    def test_native_spec_sets_ml_host(self, saved_config, tmp_path, monkeypatch):
        self._fake_native(monkeypatch, tmp_path)
        saved_config["ml_host"] = BIND_LOOPBACK
        cmd, cwd, env = m._native_ml_spec(saved_config, {})
        assert env["ML_HOST"] == BIND_LOOPBACK

    def test_venv_spec_sets_ml_host(self, saved_config, tmp_path):
        ml_dir = tmp_path / "ml"
        (ml_dir / "venv" / "bin").mkdir(parents=True)
        python = ml_dir / "venv" / "bin" / "python3"
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        saved_config["ml_dir"] = str(ml_dir)
        saved_config["ml_host"] = BIND_LOOPBACK
        cmd, cwd, env = m._venv_ml_spec(saved_config, {})
        assert env["ML_HOST"] == BIND_LOOPBACK

    def test_ml_host_defaults_to_every_interface(self, saved_config):
        assert m._ml_host(saved_config) == BIND_ANY


class TestSetupKeepsTheNewKeys:
    def test_a_repair_run_does_not_erase_the_bind_addresses(self):
        # setup rebuilds config.json from scratch and saves it wholesale, so a
        # key absent from this list is erased on every re-run.
        assert "dashboard_host" in m._PRESERVED_CONFIG_KEYS
        assert "ml_host" in m._PRESERVED_CONFIG_KEYS


class TestWatchSurvivesABrokenConfig:
    def test_cmd_watch_retries_instead_of_dying(self, monkeypatch):
        """A hand-edited config.json must not become a launchd crash loop."""
        calls = []

        def flaky_load_config():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return {"worker": False, "ml": False}

        monkeypatch.setattr(m, "_move_onto_launcher", lambda *a, **k: None)
        monkeypatch.setattr(m, "load_config", flaky_load_config)
        monkeypatch.setattr(m.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(m, "_component_enabled", lambda name, config: False)
        monkeypatch.setattr(m, "_watch_without_worker", lambda config: None)

        m.cmd_watch(argparse.Namespace())
        assert len(calls) == 3
