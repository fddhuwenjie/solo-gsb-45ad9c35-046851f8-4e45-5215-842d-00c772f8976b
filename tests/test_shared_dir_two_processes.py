"""跨进程共享数据目录的定向验证（两个真实 uvicorn 进程 + 同一数据目录）。

与 ``test_revisable_worksheets.py`` / ``test_step_result_bindings.py`` 中
基于线程池的同进程并发测试互补，本模块实际拉起两个独立的 uvicorn 工作
进程（``api-a``/``api-b``），让它们的 ``UNIT_API_DATA_DIR`` 指向同一个
临时目录，端到端验证 app/locking.py 的跨进程 flock 互斥与按磁盘签名同步：

- 事先发布的同一份定义集（含活动别名）两进程启动后都能解析；
- 一方创建的可修订工作单，另一方立即能按标识读取（输入、逐步结果、
  修订号一致）并继续修订；
- 两进程（各自再带并发线程）同时以相同 expected_revision 提交 PATCH：
  至多一笔成功，其余全部 409 且响应带回最新修订号；最终落盘结果恰好
  只包含一次修订，其他工作单不被任何一方的旧缓存覆盖；
- 上下游绑定可跨进程建立：在一进程建上游、另一进程绑下游，再回第一进程
  修订上游，两单读到的都是配套的输入、逐步结果与关联修订号；联动重算
  失败（422）时两单保持原样；无关联工作单照常使用。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest


# ---------------------------------------------------------------------------
# 两个真实 API 进程（共享同一数据目录）
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(base: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                return
        except Exception as exc:  # 进程尚未开始监听
            last_exc = exc
        time.sleep(0.1)
    raise RuntimeError(f"API {base} 未在 {timeout}s 内就绪: {last_exc!r}")


def _seed_definitions(data_dir: str) -> tuple[str, str]:
    """在两进程启动前往共享数据目录发布一份定义集并建立活动别名。

    直接在第三个一次性进程里调用存储层（不经 HTTP），等价于“事先发布”：
    两个 API 进程启动加载时即可见到同一 ``custom_unit_set_id`` 与别名。
    返回 (定义集标识, 别名)。
    """

    script = (
        "import sys\n"
        "from decimal import Decimal\n"
        "from app import unit_sets\n"
        "from app.schemas import CustomUnitItem\n"
        "store = unit_sets.UnitSetStore(sys.argv[1])\n"
        "store.load()\n"
        "set_id = store.publish([CustomUnitItem(symbol='ft', factor=Decimal('0.3048'), base_unit='m')])\n"
        "store.bind_alias('lengths', set_id, None)\n"
        "print(set_id)\n"
    )
    env = _server_env(data_dir)
    proc = subprocess.run(
        [sys.executable, "-c", script, data_dir],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip(), "lengths"


def _server_env(data_dir: str) -> dict[str, str]:
    """子进程环境：把本解释器的第三方包目录（若在 PYTHONPATH 中）一并继承。"""

    env = dict(os.environ, UNIT_API_DATA_DIR=data_dir)
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return env


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    """启动共享同一数据目录的两个 API 进程；返回两进程的 HTTP base URL。"""

    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = str(tmp_path_factory.mktemp("shared-data"))
    set_id, alias = _seed_definitions(data_dir)

    port_a, port_b = _free_port(), _free_port()
    env = _server_env(data_dir)
    common = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1"]
    procs = [
        subprocess.Popen(
            common + ["--port", str(port_a)],
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ),
        subprocess.Popen(
            common + ["--port", str(port_b)],
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ),
    ]
    bases = [f"http://127.0.0.1:{port_a}", f"http://127.0.0.1:{port_b}"]
    try:
        for base in bases:
            _wait_healthy(base)
        yield {
            "a": bases[0],
            "b": bases[1],
            "set_id": set_id,
            "alias": alias,
            "data_dir": data_dir,
        }
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def http(method, base, path, **kwargs):
    return httpx.request(method, f"{base}{path}", timeout=10, **kwargs)


def standard_body(set_id, **overrides):
    body = {
        "variables": {
            "v": {"value": "60", "unit": "km/h"},
            "t": {"value": "2", "unit": "h"},
        },
        "steps": [
            {"name": "s", "formula": "v*t", "output_unit": "km"},
            {"name": "d", "equation": "v*th = s", "solve_for": "th",
             "output_unit": "h"},
        ],
        "custom_unit_set_id": set_id,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# 事先发布的同一份定义集：两进程都能解析
# ---------------------------------------------------------------------------


def test_prepublished_set_and_alias_visible_in_both_processes(cluster):
    sid = cluster["set_id"]
    for base in (cluster["a"], cluster["b"]):
        # 标识直引：两进程都能用同一已发布定义集完成换算
        r = http(
            "POST", base, "/convert",
            json={"value": "1", "from": "ft", "to": "m",
                  "custom_unit_set_id": sid},
        )
        assert r.status_code == 200, r.text
        assert r.json()["value"] == "0.3048"
        # 活动别名同样在两进程可见且指向同一标识
        r = http("GET", base, f"/custom-unit-set-aliases/{cluster['alias']}")
        assert r.status_code == 200
        assert r.json()["custom_unit_set_id"] == sid


def test_set_published_in_one_process_immediately_visible_in_other(cluster):
    r = http(
        "POST", cluster["a"], "/custom-unit-sets",
        json={"custom_units": [
            {"symbol": "span", "factor": "2.5", "base_unit": "m"}]},
    )
    assert r.status_code == 201, r.text
    new_id = r.json()["id"]
    # 进程 B 不重启，立即就能引用进程 A 刚发布的定义集（无旧缓存遮蔽）
    r = http(
        "POST", cluster["b"], "/convert",
        json={"value": "2", "from": "span", "to": "m",
              "custom_unit_set_id": new_id},
    )
    assert r.status_code == 200, r.text
    assert r.json()["value"] == "5"


# ---------------------------------------------------------------------------
# 一方创建，另一方立即可读、可修订
# ---------------------------------------------------------------------------


def test_worksheet_created_in_one_process_readable_and_revisable_in_other(cluster):
    sid = cluster["set_id"]
    r = http("POST", cluster["a"], "/revisable-worksheets", json=standard_body(sid))
    assert r.status_code == 201, r.text
    created = r.json()
    wid = created["id"]
    assert created["revision"] == 1
    assert [s["value"] for s in created["results"]] == ["120", "2"]

    # 进程 B 立即读取到同一修订的输入、逐步结果与修订号
    r = http("GET", cluster["b"], f"/revisable-worksheets/{wid}")
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["revision"] == 1
    assert got["variables"] == created["variables"]
    assert got["results"] == created["results"]
    assert got["custom_unit_set_id"] == sid

    # 进程 B 直接在该单上修订成功
    r = http(
        "PATCH", cluster["b"], f"/revisable-worksheets/{wid}",
        json={"expected_revision": 1,
              "variable": {"name": "t", "value": "3", "unit": "h"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 2
    assert [s["value"] for s in r.json()["results"]] == ["180", "3"]

    # 进程 A 立即读到 B 的修订结果
    r = http("GET", cluster["a"], f"/revisable-worksheets/{wid}")
    assert r.status_code == 200
    body = r.json()
    assert body["revision"] == 2
    assert body["variables"]["t"] == {"value": "3", "unit": "h"}
    assert [s["value"] for s in body["results"]] == ["180", "3"]


# ---------------------------------------------------------------------------
# 跨进程同一 expected_revision 并发提交：至多一笔成功，其余 409 带最新号
# ---------------------------------------------------------------------------


def test_cross_process_concurrent_patches_same_revision_only_one_wins(cluster):
    sid = cluster["set_id"]
    wid = http(
        "POST", cluster["a"], "/revisable-worksheets", json=standard_body(sid)
    ).json()["id"]
    # 让两进程都先把该单加载进各自缓存（陈旧但 revision 相同）
    assert http("GET", cluster["a"], f"/revisable-worksheets/{wid}").status_code == 200
    assert http("GET", cluster["b"], f"/revisable-worksheets/{wid}").status_code == 200

    def attempt(base, i):
        r = http(
            "PATCH", base, f"/revisable-worksheets/{wid}",
            json={"expected_revision": 1,
                  "variable": {"name": "t", "value": str(10 + i), "unit": "h"}},
        )
        return r.status_code, (r.json() if r.content else {})

    # 两进程各 8 个并发请求，全部基于 revision=1
    with ThreadPoolExecutor(16) as pool:
        outcomes = list(pool.map(
            lambda args: attempt(args[0], args[1]),
            [(cluster["a"], i) for i in range(8)]
            + [(cluster["b"], i) for i in range(8, 16)],
        ))
    statuses = [code for code, _ in outcomes]
    assert statuses.count(200) == 1, statuses
    assert statuses.count(409) == 15, statuses

    # 每个 409 都带回最新修订号（2），且错误码为 revision_conflict
    for code, body in outcomes:
        if code == 409:
            assert body["detail"]["error"] == "revision_conflict"
            assert body["detail"]["current_revision"] == 2

    # 最终落盘只推进了一次修订：两进程读到同一 revision=2 与同一份结果
    for base in (cluster["a"], cluster["b"]):
        body = http("GET", base, f"/revisable-worksheets/{wid}").json()
        assert body["revision"] == 2
        assert len(body["results"]) == 2
        assert body["results"][1]["name"] == "d"

    # 旧 revision 的迟到请求仍为 409 且带回最新号；按新号重试成功
    r = http(
        "PATCH", cluster["a"], f"/revisable-worksheets/{wid}",
        json={"expected_revision": 1,
              "variable": {"name": "t", "value": "4", "unit": "h"}},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["current_revision"] == 2
    r = http(
        "PATCH", cluster["b"], f"/revisable-worksheets/{wid}",
        json={"expected_revision": 2,
              "variable": {"name": "t", "value": "4", "unit": "h"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 3


# ---------------------------------------------------------------------------
# 其他工作单不被旧缓存覆盖
# ---------------------------------------------------------------------------


def test_unrelated_worksheet_created_elsewhere_is_not_clobbered(cluster):
    sid = cluster["set_id"]
    # 先在 B 上创建并修订一单，使 B 的缓存里持有旧版本内容；再让 A 与 B
    # 交错各自创建不同的新单——任何一方落盘时若基于旧缓存全量写回，就会
    # 丢掉另一方刚创建的工作单（或把种子单的修订号回卷）。
    first = http(
        "POST", cluster["b"], "/revisable-worksheets", json=standard_body(sid)
    ).json()
    wid_b_seed = first["id"]
    assert http(
        "PATCH", cluster["b"], f"/revisable-worksheets/{wid_b_seed}",
        json={"expected_revision": 1,
              "variable": {"name": "t", "value": "3", "unit": "h"}},
    ).status_code == 200

    created_ids = []
    # 交错：A 建一单 -> B 建一单 -> A 再建一单 -> B 再建一单
    for base in (cluster["a"], cluster["b"], cluster["a"], cluster["b"]):
        r = http(
            "POST", base, "/revisable-worksheets", json=standard_body(sid)
        )
        assert r.status_code == 201, r.text
        created_ids.append(r.json()["id"])

    all_ids = created_ids + [wid_b_seed]
    # 通过两个进程的 GET 交叉核对：每张单在任一进程都按最新修订可见，
    # 说明每次落盘都包含了此前全部工作单（无旧缓存覆盖）
    for wid in all_ids:
        for base in (cluster["a"], cluster["b"]):
            r = http("GET", base, f"/revisable-worksheets/{wid}")
            assert r.status_code == 200, (wid, base, r.text)

    # 种子单仍是 revision=2，未被任何创建操作回卷为 1
    body = http("GET", cluster["a"],
                f"/revisable-worksheets/{wid_b_seed}").json()
    assert body["revision"] == 2
    assert body["variables"]["t"] == {"value": "3", "unit": "h"}

    # 落盘文件本身也完整保留全部工作单
    with open(os.path.join(cluster["data_dir"], "revisable_worksheets.json"),
              encoding="utf-8") as f:
        persisted = json.load(f)
    assert {wid_b_seed, *created_ids} <= set(persisted["sheets"])


# ---------------------------------------------------------------------------
# 跨进程上下游绑定与联动
# ---------------------------------------------------------------------------


def test_cross_process_binding_and_propagation(cluster):
    sid = cluster["set_id"]
    # 上游建立在进程 A
    up = http(
        "POST", cluster["a"], "/revisable-worksheets", json=standard_body(sid)
    ).json()
    upstream_id = up["id"]

    # 进程 B 立即能读到上游，并在 B 上把上游步骤 s 绑定为新下游的变量
    downstream_payload = {
        "variables": {
            "s": {"value": "0", "unit": "h"},
            "k": {"value": "2", "unit": "1"},
        },
        "steps": [
            {"name": "half", "formula": "s/2", "output_unit": "km"},
            {"name": "again", "formula": "half*k", "output_unit": "km"},
        ],
    }
    r = http(
        "POST", cluster["b"],
        f"/revisable-worksheets/{upstream_id}/bindings",
        json={"step_name": "s", "expected_revision": 1, **downstream_payload},
    )
    assert r.status_code == 201, r.text
    down = r.json()
    downstream_id = down["id"]
    assert down["revision"] == 1
    assert down["variables"]["s"] == {"value": "120", "unit": "km"}
    assert [x["value"] for x in down["results"]] == ["60", "120"]
    assert down["binding"] == {
        "upstream_worksheet_id": upstream_id,
        "step_name": "s",
        "upstream_revision": 1,
    }

    # 两进程读两单都得到配套内容（关联修订号一致）
    for base in (cluster["a"], cluster["b"]):
        u = http("GET", base, f"/revisable-worksheets/{upstream_id}").json()
        d = http("GET", base, f"/revisable-worksheets/{downstream_id}").json()
        assert "binding" not in u
        assert d["binding"]["upstream_worksheet_id"] == upstream_id
        assert d["binding"]["upstream_revision"] == u["revision"] == 1
        assert d["variables"]["s"]["value"] == u["results"][0]["value"] == "120"

    # 回到进程 A 修订上游：两单联动共同落盘
    r = http(
        "PATCH", cluster["a"], f"/revisable-worksheets/{upstream_id}",
        json={"expected_revision": 1,
              "variable": {"name": "t", "value": "3", "unit": "h"}},
    )
    assert r.status_code == 200, r.text
    assert [x["value"] for x in r.json()["results"]] == ["180", "3"]

    for base in (cluster["a"], cluster["b"]):
        u = http("GET", base, f"/revisable-worksheets/{upstream_id}").json()
        d = http("GET", base, f"/revisable-worksheets/{downstream_id}").json()
        assert u["revision"] == 2
        assert d["revision"] == 2
        assert d["variables"]["s"] == {"value": "180", "unit": "km"}
        assert [x["value"] for x in d["results"]] == ["90", "180"]
        assert d["binding"]["upstream_revision"] == 2

    # 进程 B 上用过期的上游修订号尝试联动修订 -> 409 带最新号，两单不变
    r = http(
        "PATCH", cluster["b"], f"/revisable-worksheets/{upstream_id}",
        json={"expected_revision": 1,
              "variable": {"name": "t", "value": "4", "unit": "h"}},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["current_revision"] == 2
    d = http("GET", cluster["b"], f"/revisable-worksheets/{downstream_id}").json()
    assert d["revision"] == 2 and d["variables"]["s"]["value"] == "180"

    # 下游自治（在 B 上改非绑定变量），绑定信息仍指向上游修订 2
    r = http(
        "PATCH", cluster["b"], f"/revisable-worksheets/{downstream_id}",
        json={"expected_revision": 2,
              "variable": {"name": "k", "value": "3", "unit": "1"}},
    )
    assert r.status_code == 200, r.text
    assert [x["value"] for x in r.json()["results"]] == ["90", "270"]
    assert r.json()["binding"]["upstream_revision"] == 2

    # 直接改绑定变量：422，两单均不变
    r = http(
        "PATCH", cluster["a"], f"/revisable-worksheets/{downstream_id}",
        json={"expected_revision": 3,
              "variable": {"name": "s", "value": "1", "unit": "km"}},
    )
    assert r.status_code == 422


def test_cross_process_propagation_failure_leaves_both_sheets_intact(cluster):
    sid = cluster["set_id"]
    up_body = {
        "variables": {"x": {"value": "10", "unit": "ft"}},
        "steps": [
            {"name": "a", "formula": "x", "output_unit": "m"},
            {"name": "b", "formula": "x", "output_unit": "ft"},
        ],
        "custom_unit_set_id": sid,
    }
    up = http("POST", cluster["b"], "/revisable-worksheets", json=up_body).json()
    r = http(
        "POST", cluster["a"],
        f"/revisable-worksheets/{up['id']}/bindings",
        json={
            "step_name": "b",
            "expected_revision": 1,
            "variables": {"b": {"value": "1", "unit": "ft"}},
            "steps": [
                {"name": "a2", "formula": "b*b", "output_unit": "ft^2"},
                {"name": "b2", "formula": "a2", "output_unit": "ft^2"},
            ],
        },
    )
    assert r.status_code == 201, r.text
    downstream_id = r.json()["id"]

    # 上游把 b 改成无量纲：上游可算，但联动下游 b*b -> ft^2 量纲冲突
    r = http(
        "PATCH", cluster["a"], f"/revisable-worksheets/{up['id']}",
        json={"expected_revision": 1,
              "step": {"name": "b", "formula": "1", "output_unit": "1"}},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "dimension_mismatch"

    # 两单在两个进程里都保持原样
    for base in (cluster["a"], cluster["b"]):
        u = http("GET", base, f"/revisable-worksheets/{up['id']}").json()
        d = http("GET", base, f"/revisable-worksheets/{downstream_id}").json()
        assert u["revision"] == 1 and d["revision"] == 1
        assert u["results"][1] == {"name": "b", "value": "10", "unit": "ft",
                                   "dimension": "m"}
        assert d["variables"]["b"] == {"value": "10", "unit": "ft"}
        assert [x["value"] for x in d["results"]] == ["100", "100"]

    # 失败后仍可按原修订号成功修订，联动正常完成
    r = http(
        "PATCH", cluster["b"], f"/revisable-worksheets/{up['id']}",
        json={"expected_revision": 1,
              "variable": {"name": "x", "value": "20", "unit": "ft"}},
    )
    assert r.status_code == 200, r.text
    d = http("GET", cluster["a"], f"/revisable-worksheets/{downstream_id}").json()
    assert d["revision"] == 2
    assert d["variables"]["b"] == {"value": "20", "unit": "ft"}
    assert [x["value"] for x in d["results"]] == ["400", "400"]
    assert d["binding"]["upstream_revision"] == 2
