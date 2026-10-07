"""两个真实 uvicorn 进程共享同一数据目录的端到端 HTTP 定向验证。

本测试用 ``uvicorn`` 子进程各起一个 API 实例（A:8000 系列端口、
B:8010 系列端口，具体端口由空闲端口分配），两个进程的
``UNIT_API_DATA_DIR`` 指向同一个临时目录，随后仅通过 HTTP 验证：

1. 在 A 事先发布定义集，B 立即可凭该标识创建工作单；
2. A 创建工作单，B 立即能 GET 并 PATCH（跨进程立即可读可改）；
3. A、B 并发用相同 expected_revision 提交 PATCH：恰有一笔 200，
   其余 409 且响应带回当前最新修订号；双方之后读到同一修订，且另一张
   工作单没有被旧缓存覆盖；
4. 在 B 建立上游（A 创建）的下游绑定，A 修订上游后，两进程 GET 两单
   都得到配套的输入、逐步结果与关联修订号；
5. 让联动下游重算失败（422）时，两个进程读到两单均保持原修订；
6. 无关联工作单在两进程间照常读取与修订。

环境中无法启动 uvicorn 子进程（如未安装运行依赖或端口不可用）时
整个模块自动跳过——核心语义另有
``test_shared_dir_multiprocess.py`` 的存储级确定性测试覆盖。
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("UNIT_API_RUN_DUAL_PROCESS_TESTS") != "1",
    reason=(
        "真实双进程 HTTP 测试默认跳过（设置 "
        "UNIT_API_RUN_DUAL_PROCESS_TESTS=1 启用）；存储级语义见 "
        "test_shared_dir_multiprocess.py"
    ),
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_TIMEOUT = 20.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _api_process(data_dir: Path, port: int):
    env = dict(os.environ)
    env["UNIT_API_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                pytest.skip(f"uvicorn 进程提前退出:\n{output[-2000:]}")
            try:
                if httpx.get(f"{base}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        else:
            proc.terminate()
            pytest.skip("uvicorn 在超时时间内未就绪")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("shared-data")
    port_a, port_b = _free_port(), _free_port()
    with (
        _api_process(data_dir, port_a) as base_a,
        _api_process(data_dir, port_b) as base_b,
    ):
        yield {
            "a": httpx.Client(base_url=base_a, timeout=10),
            "b": httpx.Client(base_url=base_b, timeout=10),
            "dir": data_dir,
        }


def _upstream(set_id, *, t="2"):
    return {
        "variables": {
            "v": {"value": "60", "unit": "km/h"},
            "t": {"value": t, "unit": "h"},
        },
        "steps": [
            {"name": "s", "formula": "v*t", "output_unit": "km"},
            {"name": "th", "equation": "v*th = s", "solve_for": "th",
             "output_unit": "h"},
        ],
        "custom_unit_set_id": set_id,
    }


def _publish(client, symbol="ft", factor="0.3048", base_unit="m"):
    r = client.post(
        "/custom-unit-sets",
        json={"custom_units": [
            {"symbol": symbol, "factor": factor, "base_unit": base_unit}
        ]},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _create(client, set_id, *, t="2"):
    r = client.post("/revisable-worksheets", json=_upstream(set_id, t=t))
    assert r.status_code == 201, r.text
    return r.json()


def _get(client, wid):
    r = client.get(f"/revisable-worksheets/{wid}")
    assert r.status_code == 200, r.text
    return r.json()


def _patch(client, wid, expected, **change):
    return client.patch(
        f"/revisable-worksheets/{wid}",
        json={"expected_revision": expected, **change},
    )


def test_prepublished_unit_set_shared_and_worksheet_immediately_visible(cluster):
    a, b = cluster["a"], cluster["b"]
    set_id = _publish(a)  # A 事先发布

    # B 立即能用同一标识创建工作单
    r = b.post("/revisable-worksheets", json=_upstream(set_id))
    assert r.status_code == 201, r.text
    wid = r.json()["id"]

    # A 创建一张，B 立即读到
    other = _create(a, set_id)
    got_b = _get(b, other["id"])
    assert got_b["revision"] == 1
    assert [s["value"] for s in got_b["results"]] == ["120", "2"]

    # B 修订 A 创建的单，A 立即读到 B 的修订
    r = _patch(b, other["id"], 1,
               variable={"name": "t", "value": "3", "unit": "h"})
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 2
    got_a = _get(a, other["id"])
    assert got_a["revision"] == 2
    assert got_a["variables"]["t"]["value"] == "3"


def test_concurrent_same_expected_revision_only_one_200_and_other_sheet_intact(
    cluster,
):
    a, b = cluster["a"], cluster["b"]
    set_id = _get_shared_set_id(cluster)

    target = _create(a, set_id)["id"]
    other = _create(b, set_id, t="2")["id"]

    def attempt(client, value):
        r = _patch(client, target, 1,
                   variable={"name": "t", "value": value, "unit": "h"})
        return r.status_code, r.json()

    # 两侧并发以 expected_revision=1 提交（各发多笔放大交叠）
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        futs = []
        for i in range(4):
            futs.append(pool.submit(attempt, a, str(3 + i)))
            futs.append(pool.submit(attempt, b, str(7 + i)))
        outcomes = [f.result() for f in futs]

    statuses = [code for code, _ in outcomes]
    assert statuses.count(200) == 1
    assert all(code == 409 for code in statuses if code != 200)
    # 每笔 409 都带回服务端最新修订号
    for code, body in outcomes:
        if code == 409:
            assert body["detail"]["error"] == "revision_conflict"
            assert body["detail"]["current_revision"] == 2

    # 两个进程读到同一修订，且结果是唯一成功者的输入
    ga, gb = _get(a, target), _get(b, target)
    assert ga["revision"] == gb["revision"] == 2
    assert ga["variables"]["t"]["value"] == gb["variables"]["t"]["value"]
    assert [s["value"] for s in ga["results"]] == [
        s["value"] for s in gb["results"]
    ]
    # 另一张工作单未被任何一方的旧缓存覆盖
    assert _get(a, other)["revision"] == 1
    assert _get(b, other)["revision"] == 1


def test_upstream_downstream_binding_coherent_across_processes(cluster):
    a, b = cluster["a"], cluster["b"]
    set_id = _get_shared_set_id(cluster)

    # 上游由 A 创建
    up = _create(a, set_id)
    up_id = up["id"]
    # 下游在 B 上建立：绑定上游步骤 s
    r = b.post(
        f"/revisable-worksheets/{up_id}/bindings",
        json={
            "step_name": "s",
            "expected_revision": 1,
            "variables": {
                "s": {"value": "0", "unit": "h"},
                "k": {"value": "2", "unit": "1"},
            },
            "steps": [
                {"name": "half", "formula": "s/2", "output_unit": "km"},
                {"name": "again", "formula": "half*k", "output_unit": "km"},
            ],
        },
    )
    assert r.status_code == 201, r.text
    down_id = r.json()["id"]

    # 任一方读取两单都配套
    for client in (a, b):
        u, d = _get(client, up_id), _get(client, down_id)
        assert u["revision"] == d["binding"]["upstream_revision"] == 1
        step_s = next(s for s in u["results"] if s["name"] == "s")
        assert d["variables"]["s"] == {"value": step_s["value"],
                                      "unit": step_s["unit"]}

    # A 修订上游 2 h -> 3 h；两单联动共同提交
    r = _patch(a, up_id, 1,
               variable={"name": "t", "value": "3", "unit": "h"})
    assert r.status_code == 200, r.text

    for client in (a, b):
        u, d = _get(client, up_id), _get(client, down_id)
        assert u["revision"] == 2 and d["revision"] == 2
        assert next(s for s in u["results"] if s["name"] == "s")["value"] == "180"
        assert d["variables"]["s"] == {"value": "180", "unit": "km"}
        assert [s["value"] for s in d["results"]] == ["90", "180"]
        assert d["binding"]["upstream_revision"] == 2


def test_downstream_failure_leaves_both_sheets_unchanged(cluster):
    a, b = cluster["a"], cluster["b"]
    set_id = _get_shared_set_id(cluster)
    up_id = a.post(
        "/revisable-worksheets",
        json={
            "variables": {"x": {"value": "10", "unit": "ft"}},
            "steps": [
                {"name": "p", "formula": "x", "output_unit": "m"},
                {"name": "q", "formula": "x", "output_unit": "ft"},
            ],
            "custom_unit_set_id": set_id,
        },
    ).json()["id"]
    r = b.post(
        f"/revisable-worksheets/{up_id}/bindings",
        json={
            "step_name": "q",
            "expected_revision": 1,
            "variables": {"q": {"value": "1", "unit": "ft"}},
            "steps": [
                {"name": "q2", "formula": "q*q", "output_unit": "ft^2"},
                {"name": "q3", "formula": "q2", "output_unit": "ft^2"},
            ],
        },
    )
    assert r.status_code == 201, r.text
    down_id = r.json()["id"]

    # 上游 q 改无量纲：上游可算，下游 q*q->ft^2 失败 -> 422，两单都不变
    r = _patch(a, up_id, 1,
               step={"name": "q", "formula": "1", "output_unit": "1"})
    assert r.status_code == 422
    for client in (a, b):
        assert _get(client, up_id)["revision"] == 1
        d = _get(client, down_id)
        assert d["revision"] == 1
        assert [s["value"] for s in d["results"]] == ["100", "100"]
        assert d["binding"]["upstream_revision"] == 1


def test_unbound_worksheet_works_normally_across_processes(cluster):
    a, b = cluster["a"], cluster["b"]
    set_id = _get_shared_set_id(cluster)
    wid = _create(a, set_id)["id"]
    assert "binding" not in _get(a, wid) and "binding" not in _get(b, wid)
    r = _patch(b, wid, 1,
               variable={"name": "t", "value": "4", "unit": "h"})
    assert r.status_code == 200 and "binding" not in r.json()
    assert _get(a, wid)["revision"] == 2


def _get_shared_set_id(cluster):
    """所有进程共用一份事先发布的定义集（每个模块运行发布一次即可）。"""

    cache = cluster.setdefault("_set_id", None)
    if cache is None:
        cache = _publish(cluster["a"], symbol="ft", factor="0.3048",
                         base_unit="m")
        cluster["_set_id"] = cache
    return cache
