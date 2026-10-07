from fastapi.testclient import TestClient
from conftest import AGENTIC, JS_ONLY

import api


def test_screen_and_results_endpoints(tmp_path, monkeypatch):
    folder = tmp_path / "resumes"
    folder.mkdir()
    (folder / "a.txt").write_text(AGENTIC, encoding="utf-8")
    (folder / "b.txt").write_text(JS_ONLY, encoding="utf-8")
    monkeypatch.setattr(api, "OUTPUT", tmp_path / "out" / "results.json")
    monkeypatch.setenv("CACHE_ENABLED", "false")
    api._state.clear()
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/screen", json={"input_dir": str(folder), "use_llm": False, "use_github": False})
    assert r.status_code == 200
    body = r.json()
    assert body["batch_summary"]["eligible"] == 1 and body["batch_summary"]["rejected"] == 1
    assert body["ranked_candidates"][0]["rank"] == 1
    assert client.get("/results").json() == body
    assert (tmp_path / "out" / "results.json").exists()
    assert client.post("/screen", json={"input_dir": str(tmp_path / "missing")}).status_code == 400
