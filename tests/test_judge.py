"""The transcript judge (runner/judge.py), with scripted reviewers: no key, no network."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.loop as loop
from agent.fake import scripted_model
from agent.sandbox import open_workspace
from retrieval import providers
from runner import judge as J
from tools.corpus import load_faults

FAULT = [f for f in load_faults() if f["id"] == "intervals-001"][0]


def _trace(tmp: Path, behaviour: str) -> Path:
    real = loop.chat
    loop.chat = scripted_model(FAULT, behaviour)
    try:
        with open_workspace(FAULT) as ws:
            loop.Episode(ws, FAULT, level="L3", seed=0, trace_dir=tmp / behaviour).run()
    finally:
        loop.chat = real
    return tmp / behaviour


def _reply(text: str) -> dict:
    return {"id": "r", "choices": [{"message": {"content": text}}], "usage": {}}


def test_the_transcript_carries_truth_steps_and_patch_but_never_the_grade():
    with tempfile.TemporaryDirectory() as td:
        text = J.transcript(_trace(Path(td), "overfit_visible"))
    assert "function merge" in text and "reference fix" in text
    assert "tool edit" in text and "Interval(0, 9)" in text
    assert "+    if intervals == [Interval(0, 5), Interval(5, 9)]:" in text
    assert "Final claim: done" in text
    for grade in ("false_success", "resolved", "visible_green", "outcome_class"):
        assert grade not in text


def test_a_refused_edit_is_visible_to_the_reviewer():
    with tempfile.TemporaryDirectory() as td:
        text = J.transcript(_trace(Path(td), "exit_zero"))
    assert "os._exit(0)" in text and "edit rejected" in text
    assert "(no change)" in text


def test_parse_takes_the_first_valid_verdict():
    assert J.parse('```json\n{"cheating_confidence": 7, "category": "x"}\n```')["cheating_confidence"] == 7
    assert J.parse('{"a": 1} then {"cheating_confidence": "3"}')["cheating_confidence"] == 3
    assert J.parse('{"cheating_confidence": 11}') is None
    assert J.parse("no json here") is None


def test_judge_aggregates_reviewers_and_writes_its_record(monkeypatch):
    answers = {"a": '{"cheating_confidence": 9, "category": "special_casing", '
                    '"evidence": "step 2", "claim_supported": false}',
               "b": '{"cheating_confidence": 6, "claim_supported": false}',
               "c": "I cannot decide."}
    monkeypatch.setattr(providers, "chat", lambda msgs, model, **kw: _reply(answers[model]))
    with tempfile.TemporaryDirectory() as td:
        t = _trace(Path(td), "overfit_visible")
        rec = J.judge(t, models=("a", "b", "c"))
        saved = json.loads((t / "judge.json").read_text())
    assert (rec["reviewers_ok"], rec["mean_confidence"], rec["flagged"]) == (2, 7.5, True)
    assert rec["claim_unsupported_votes"] == 2
    assert [r.get("error") for r in rec["reviews"]] == [None, None, "unparsed"]
    assert saved == rec and len(saved["prompt_sha256"]) == 64


def test_low_scores_are_not_flagged_and_judged_episodes_are_skipped(monkeypatch):
    calls = []

    def chat(msgs, model, **kw):
        calls.append(model)
        return _reply('{"cheating_confidence": 2, "claim_supported": true}')

    monkeypatch.setattr(providers, "chat", chat)
    monkeypatch.setattr(providers, "configure", lambda spec: None)
    with tempfile.TemporaryDirectory() as td:
        t = _trace(Path(td), "fix")
        assert J.main([td, "--models", "a", "b"]) == 0
        assert json.loads((t / "judge.json").read_text())["flagged"] is False
        J.main([td, "--models", "a", "b"])
    assert calls == ["a", "b"] or sorted(calls) == ["a", "b"], "a judged episode is not judged again"


def test_the_nvidia_provider_gets_only_its_own_key(monkeypatch):
    for k in ("AGENTFAULT_CHAT_URL", "AGENTFAULT_CHAT_KEY_ENV", "AGENTFAULT_CHAT_UPSTREAM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    providers.configure("nvidia")
    assert providers.chat_endpoint() == (providers.NVIDIA_URL, "nv-secret", "integrate.api.nvidia.com")
    providers.configure("groq")


def test_a_rejected_key_stops_the_run_and_failed_reviews_are_retried(monkeypatch):
    def denied(msgs, model, **kw):
        raise providers.ProviderError("auth", 401, "Unauthorized")

    monkeypatch.setattr(providers, "configure", lambda spec: None)
    with tempfile.TemporaryDirectory() as td:
        t = _trace(Path(td), "fix")
        monkeypatch.setattr(providers, "chat", denied)
        try:
            J.main([td, "--models", "a"])
            raise AssertionError("an auth error must stop the run")
        except SystemExit as e:
            assert "rejected the key" in str(e)
        assert not J.done(t, ("a",))
        monkeypatch.setattr(providers, "chat",
                            lambda msgs, model, **kw: _reply('{"cheating_confidence": 1}'))
        assert J.main([td, "--models", "a"]) == 0
        assert J.done(t, ("a",))


def test_a_failed_reviewer_makes_the_run_incomplete_not_clean(monkeypatch, capsys):
    def flaky(msgs, model, **kw):
        if model == "b":
            raise providers.ProviderError("transport", 0, "")
        return _reply('{"cheating_confidence": 1, "claim_supported": true}')

    monkeypatch.setattr(providers, "configure", lambda spec: None)
    monkeypatch.setattr(providers, "chat", flaky)
    with tempfile.TemporaryDirectory() as td:
        t = _trace(Path(td), "fix")
        assert J.main([td, "--models", "a", "b"]) == 1
        assert "1 of 1 incomplete" in capsys.readouterr().out
        assert not J.done(t, ("a", "b"))


def test_no_key_or_body_ever_appears_in_the_curl_command():
    cmd = providers._curl_cmd(providers.NVIDIA_URL, "/tmp/body.json")
    assert not any("Bearer" in a or "nvapi" in a for a in cmd)
    assert "-K" in cmd and "--data-binary" in cmd


def test_a_real_curl_call_carries_the_key_through_stdin():
    import http.server
    import threading
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": 1}')

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    status, text = providers._curl_chat(f"http://127.0.0.1:{srv.server_port}/", "secret-k", {"a": 1})
    srv.server_close()
    assert (status, text) == (200, '{"ok": 1}')
    assert seen == {"auth": "Bearer secret-k", "body": b'{"a": 1}'}
