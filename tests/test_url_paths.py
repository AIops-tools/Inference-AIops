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
def test_drain_replica_encodes_traversal_in_requested_url():
    """Drive a real InferenceConnection with a fake client: the URL actually
    requested must carry the encoded segment, never a raw ``../``."""
    from inference_aiops.config import TargetConfig
    from inference_aiops.connection import InferenceConnection
    from inference_aiops.ops import serve as ops

    seen: dict[str, str] = {}

    class _Client:
        def request(self, method: str, url: str, **kwargs) -> _Resp:
            seen["url"] = url
            return _Resp(200, {})

        def close(self) -> None:
            pass

    conn = InferenceConnection(TargetConfig(name="t", host="gpu.local"), client=_Client())
    ops.drain_replica(conn, "app1", "dep1", "../../api/admin")
    assert "../" not in seen["url"]
    assert "..%2F..%2Fapi%2Fadmin" in seen["url"]


@pytest.mark.unit
def test_undeploy_and_scale_paths_encode_hostile_ids():
    from inference_aiops.ops import deploy as dp
    from inference_aiops.ops import serve as sv

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {"applications": {}}
    dp.undeploy_model(conn, "../jobs")
    (path,) = conn.delete_ray.call_args.args
    assert "../" not in path and "..%2Fjobs" in path

    conn2 = MagicMock(name="conn2")
    dp.redeploy_deployment(conn2, "a/b", "c/d")
    (path2,) = conn2.put_ray.call_args.args
    assert path2 == "/api/serve/applications/a%2Fb/deployments/c%2Fd/redeploy"

    conn3 = MagicMock(name="conn3")
    sv.update_autoscale_config(conn3, "../x", "dep", min_replicas=1)
    (path3,) = conn3.put_ray.call_args.args
    assert "../" not in path3 and "..%2Fx" in path3


@pytest.mark.unit
def test_cancel_job_encodes_job_id():
    from inference_aiops.ops import ray_cluster as rc

    conn = MagicMock(name="conn")
    rc.cancel_job(conn, "../serve/applications/app1")
    (path,) = conn.post_ray.call_args.args
    assert "../" not in path
    assert path.endswith("..%2Fserve%2Fapplications%2Fapp1/stop")
