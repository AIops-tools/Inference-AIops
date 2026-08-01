"""URL path-segment encoding — agent-supplied ids must not rewrite REST paths.

Every ops function that interpolates an identifier (application, deployment,
replica id, job id) into a Ray REST *path* must route it through ``_seg`` so a
hostile value like ``../admin`` stays one percent-encoded segment instead of
traversing to a different endpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class _Resp:
    def __init__(self, status: int = 200, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = ""
        self.content = b"{}"

    def json(self) -> dict:
        return self._payload


@pytest.mark.unit
def test_drain_replica_makes_no_request_no_endpoint_exists():
    """Ray Serve has no per-replica drain REST endpoint, so a hostile replica id
    cannot reach a URL at all — the op refuses before touching the transport."""
    from inference_aiops.config import TargetConfig
    from inference_aiops.connection import EngineCapabilityError, InferenceConnection
    from inference_aiops.ops import serve as ops

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, method: str, url: str, **kwargs) -> _Resp:
            self.calls += 1
            return _Resp(200, {})

        def close(self) -> None:
            pass

    client = _Client()
    conn = InferenceConnection(TargetConfig(name="t", host="gpu.local"), client=client)
    with pytest.raises(EngineCapabilityError):
        ops.drain_replica(conn, "app1", "dep1", "../../api/admin")
    assert client.calls == 0, "drain must not hit the transport at all"


@pytest.mark.unit
def test_serve_writes_carry_hostile_ids_in_the_body_not_the_url():
    """Serve writes PUT the whole declarative schema to a CONSTANT path; the app
    /deployment ids are dict keys and JSON values, never URL segments, so a
    hostile id cannot traverse to another endpoint."""
    from inference_aiops.ops import deploy as dp
    from inference_aiops.ops import serve as sv

    # deploy merges into the constant /applications/ path regardless of the name
    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {"applications": {}}
    dp.deploy_model(conn, "../jobs", "m:app")
    (path,) = conn.put_ray.call_args.args
    assert path == "/api/serve/applications/"
    body = conn.put_ray.call_args.kwargs["json"]["applications"]
    assert any(a.get("name") == "../jobs" for a in body)  # name lives in the body

    # a hostile app id that does not exist is a lookup miss, never a traversal
    from inference_aiops.connection import InferenceApiError

    conn2 = MagicMock(name="conn2")
    conn2.get_ray.return_value = {"applications": {}}
    with pytest.raises(InferenceApiError):
        sv.update_autoscale_config(conn2, "../x", "dep", min_replicas=1)
    conn2.put_ray.assert_not_called()


@pytest.mark.unit
def test_cancel_job_encodes_job_id():
    from inference_aiops.ops import ray_cluster as rc

    conn = MagicMock(name="conn")
    rc.cancel_job(conn, "../serve/applications/app1")
    (path,) = conn.post_ray.call_args.args
    assert "../" not in path
    assert path.endswith("..%2Fserve%2Fapplications%2Fapp1/stop")
