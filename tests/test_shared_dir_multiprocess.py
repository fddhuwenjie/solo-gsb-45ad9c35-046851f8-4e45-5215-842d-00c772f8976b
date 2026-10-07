"""两个 API 进程共享同一数据目录的定向验证（存储级、确定性）。

用同一临时数据目录上的两个独立 ``UnitSetStore`` / ``RevisionStore`` 实例
模拟两个互不共享内存缓存的进程（A、B 各自只在首次访问时加载磁盘，之后
主要依赖缓存 + 文件签名失效）：

- 一方事先发布的定义集，另一方立即可见并可凭以创建/计算；
- 一方创建的可修订工作单，另一方立即能读取与修订；
- 双方同时提交相同 expected_revision：至多一笔成功，其余得到带最新
  修订号的 RevisionConflictError（409），且不会用旧缓存覆盖别人的工作单；
- 跨进程建立上下游绑定后，任一方读取两单都得到配套的输入、逐步结果与
  关联修订号；跨进程修订上游后另一方可立即读到联动后的两单；
- 下游整单重算失败时，两单（跨两个“进程”的视角）都保持原样；
- 存档仍是与单进程时代相同的 JSON 文件（version/sets 或 version/sheets），
  锁文件是数据文件之外的附加文件；无关联工作单照常使用。
"""

from __future__ import annotations

import json
import os

import pytest

from app import revisions, unit_sets
from app.revisions import VariableChange
from app.schemas import CustomUnitItem, WorksheetBody, WorksheetRequest


# ---------------------------------------------------------------------------
# 夹具：同一数据目录上两个互不知情的“进程”
# ---------------------------------------------------------------------------


@pytest.fixture
def peers(tmp_path):
    data_dir = str(tmp_path / "data")
    sets_a = unit_sets.UnitSetStore(data_dir)
    sets_a.load()
    sets_b = unit_sets.UnitSetStore(data_dir)
    sets_b.load()
    rev_a = revisions.RevisionStore(data_dir)
    rev_a.load()
    rev_b = revisions.RevisionStore(data_dir)
    rev_b.load()

    def recalc_factory(store):
        def recalc(req: WorksheetRequest):
            from app.worksheet import run_worksheet

            return run_worksheet(req, store.build_scope(req.custom_unit_set_id))

        return recalc

    return {
        "dir": data_dir,
        "sets": (sets_a, sets_b),
        "rev": (rev_a, rev_b),
        "recalc": (recalc_factory(sets_a), recalc_factory(sets_b)),
    }


def upstream_body(set_id):
    return WorksheetRequest.model_validate(
        {
            "variables": {
                "v": {"value": "60", "unit": "km/h"},
                "t": {"value": "2", "unit": "h"},
            },
            "steps": [
                {"name": "s", "formula": "v*t", "output_unit": "km"},
                {"name": "th", "equation": "v*th = s", "solve_for": "th",
                 "output_unit": "h"},
            ],
            "custom_unit_set_id": set_id,
        }
    )


def downstream_body():
    return WorksheetBody.model_validate(
        {
            "variables": {
                "s": {"value": "0", "unit": "h"},
                "k": {"value": "2", "unit": "1"},
            },
            "steps": [
                {"name": "half", "formula": "s/2", "output_unit": "km"},
                {"name": "again", "formula": "half*k", "output_unit": "km"},
            ],
        }
    )


# ---------------------------------------------------------------------------
# 事先发布的同一单位定义集跨进程共享
# ---------------------------------------------------------------------------


def test_published_unit_set_is_visible_to_other_process_immediately(peers):
    a, b = peers["sets"]
    set_id = a.publish(_item("ft", "0.3048", "m"))
    # B 启动后尚未见过该标识；下一次读立即看到（事先发布、跨进程共享）
    assert [str(i.factor) for i in b.get(set_id)] == ["0.3048"]
    # B 也能用别名切换操作同一份定义集（写路径合并而不覆盖）
    assert b.bind_alias("lengths", set_id, None) == set_id
    assert a.get_alias("lengths") == set_id
    # A 再发布一个定义集，B 的视图里两个定义集都在（旧缓存未覆盖新内容）
    second = a.publish(_item("rod", "5.0292", "m"))
    assert b.get(second) and b.get(set_id)
    assert b.get_alias("lengths") == set_id


def _item(symbol, factor, base_unit):
    return [CustomUnitItem(symbol=symbol, factor=factor, base_unit=base_unit)]


# ---------------------------------------------------------------------------
# 工作单跨进程立即可读可改
# ---------------------------------------------------------------------------


def test_worksheet_created_by_one_process_is_readable_by_other_immediately(peers):
    a, b = peers["sets"]
    ra, rb = peers["rev"]
    ca, cb = peers["recalc"]
    set_id = a.publish(_item("ft", "0.3048", "m"))

    body = upstream_body(set_id)
    wid, created = ra.create(body, set_id, ca(body))
    assert created["revision"] == 1

    # B 立即读到 A 刚创建的工作单
    snap = rb.snapshot(wid)
    assert snap["revision"] == 1
    assert snap["custom_unit_set_id"] == set_id
    assert [r["value"] for r in snap["results"]] == ["120", "2"]

    # B 直接在 A 创建的工作单上修订；A 随后读到 B 的修订
    rec = rb.revise(wid, 1, VariableChange("t", "3", "h"), cb)
    assert rec["revision"] == 2
    snap_a = ra.snapshot(wid)
    assert snap_a["revision"] == 2
    assert snap_a["worksheet"]["variables"]["t"]["value"] == "3"
    assert [r["value"] for r in snap_a["results"]] == ["180", "3"]


def test_concurrent_same_expected_revision_only_one_wins_across_processes(peers):
    a, b = peers["sets"]
    ra, rb = peers["rev"]
    ca, cb = peers["recalc"]
    set_id = a.publish(_item("ft", "0.3048", "m"))
    body = upstream_body(set_id)
    wid, _ = ra.create(body, set_id, ca(body))

    # A、B 都基于 revision=1 提交不同修改：恰好一笔成功，另一笔 409
    outcome = {}

    def commit(rev_store, recalc, value):
        try:
            rec = rev_store.revise(
                wid, 1, VariableChange("t", value, "h"), recalc
            )
            return ("ok", rec["revision"], value)
        except revisions.RevisionConflictError as exc:
            return ("conflict", exc.current_revision, value)

    # 用线程让两个“进程”的临界区真正交叠（flock 在文件描述符上互斥，
    # 同一线程顺序执行无法制造等待）；store 内部锁与文件锁共同串行化。
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as pool:
        futs = [
            pool.submit(commit, ra, ca, "3"),
            pool.submit(commit, rb, cb, "4"),
        ]
        outcome = [f.result() for f in futs]

    oks = [o for o in outcome if o[0] == "ok"]
    conflicts = [o for o in outcome if o[0] == "conflict"]
    assert len(oks) == 1
    assert len(conflicts) == 1
    # 成功者 revision=2；落选者拿到的最新修订号也是 2
    assert oks[0][1] == 2
    assert conflicts[0][1] == 2

    # 两个进程读一致：只发生一次修改，结果值是成功者提交的那个
    for store in (ra, rb):
        snap = store.snapshot(wid)
        assert snap["revision"] == 2
        assert snap["worksheet"]["variables"]["t"]["value"] == oks[0][2]


def test_stale_cache_does_not_clobber_other_worksheets_across_processes(peers):
    a, b = peers["sets"]
    ra, rb = peers["rev"]
    ca, cb = peers["recalc"]
    set_id = a.publish(_item("ft", "0.3048", "m"))

    first = upstream_body(set_id)
    w1, _ = ra.create(first, set_id, ca(first))
    second = WorksheetRequest.model_validate(
        {
            "variables": {"x": {"value": "10", "unit": "ft"}},
            "steps": [
                {"name": "p", "formula": "x", "output_unit": "m"},
                {"name": "q", "formula": "p", "output_unit": "m"},
            ],
            "custom_unit_set_id": set_id,
        }
    )
    # w2 由 B 创建；此时 A 的缓存里没有 w2
    w2, _ = rb.create(second, set_id, cb(second))

    # A 先触碰一次 w1（缓存仍只含 w1），随后 B 修订 w2 两次，再由 A 修订 w1。
    # A 的全量落盘必须基于重读后的合并快照，不能让 w2 回退/消失。
    assert ra.snapshot(w1)["revision"] == 1
    rb.revise(w2, 1, VariableChange("x", "20", "ft"), cb)
    rb.revise(w2, 2, VariableChange("x", "30", "ft"), cb)
    ra.revise(w1, 1, VariableChange("t", "9", "h"), ca)

    assert ra.snapshot(w2)["revision"] == 3
    assert ra.snapshot(w2)["worksheet"]["variables"]["x"]["value"] == "30"
    assert rb.snapshot(w1)["revision"] == 2
    assert rb.snapshot(w1)["worksheet"]["variables"]["t"]["value"] == "9"
    assert ra.count() == rb.count() == 2


# ---------------------------------------------------------------------------
# 上下游绑定跨进程
# ---------------------------------------------------------------------------


def test_binding_created_across_processes_visible_coherently_to_either(peers):
    a, b = peers["sets"]
    ra, rb = peers["rev"]
    ca, cb = peers["recalc"]
    set_id = a.publish(_item("ft", "0.3048", "m"))

    body = upstream_body(set_id)
    up_id, _ = ra.create(body, set_id, ca(body))

    # B 作为客户端凭上游当前修订号建立绑定、创建下游
    down_id, down = rb.bind_downstream(
        up_id, 1, "s", downstream_body(), cb
    )
    assert down["revision"] == 1
    assert down["binding"] == {
        "upstream_worksheet_id": up_id,
        "step_name": "s",
        "upstream_revision": 1,
    }
    assert down["worksheet"]["variables"]["s"] == {"value": "120", "unit": "km"}
    assert [r["value"] for r in down["results"]] == ["60", "120"]

    # 任一方读取两单都得到配套内容
    for store in (ra, rb):
        up = store.snapshot(up_id)
        dn = store.snapshot(down_id)
        assert up["revision"] == 1
        assert dn["revision"] == 1
        # 下游绑定变量与上游对应步骤结果一致（值与单位）
        step = next(r for r in up["results"] if r["name"] == "s")
        assert dn["worksheet"]["variables"]["s"] == {
            "value": step["value"],
            "unit": step["unit"],
        }
        assert dn["binding"]["upstream_revision"] == up["revision"]

    # A 修订上游（2 h -> 3 h，s 120 -> 180 km），联动在跨进程排他临界区内
    rec = ra.revise(up_id, 1, VariableChange("t", "3", "h"), ca)
    assert rec["revision"] == 2
    # B 立即读到配套的两单：上游 rev2、下游 rev2，结果与绑定修订号同步
    up_b = rb.snapshot(up_id)
    dn_b = rb.snapshot(down_id)
    assert up_b["results"][0]["value"] == "180"
    assert dn_b["revision"] == 2
    assert dn_b["worksheet"]["variables"]["s"] == {"value": "180", "unit": "km"}
    assert [r["value"] for r in dn_b["results"]] == ["90", "180"]
    assert dn_b["binding"]["upstream_revision"] == 2
    # A 自己看到的与 B 完全一致
    dn_a = ra.snapshot(down_id)
    assert dn_a["revision"] == 2
    assert dn_a["binding"]["upstream_revision"] == 2


def test_downstream_calculation_failure_leaves_both_sheets_unchanged(peers):
    a, b = peers["sets"]
    ra, rb = peers["rev"]
    ca, cb = peers["recalc"]
    set_id = a.publish(_item("ft", "0.3048", "m"))

    up = WorksheetRequest.model_validate(
        {
            "variables": {"x": {"value": "10", "unit": "ft"}},
            "steps": [
                {"name": "p", "formula": "x", "output_unit": "m"},
                {"name": "q", "formula": "x", "output_unit": "ft"},
            ],
            "custom_unit_set_id": set_id,
        }
    )
    up_id, _ = ra.create(up, set_id, ca(up))
    down_body = WorksheetBody.model_validate(
        {
            "variables": {"q": {"value": "1", "unit": "ft"}},
            "steps": [
                {"name": "q2", "formula": "q*q", "output_unit": "ft^2"},
                {"name": "q3", "formula": "q2", "output_unit": "ft^2"},
            ],
        }
    )
    down_id, _ = rb.bind_downstream(up_id, 1, "q", down_body, cb)

    # 上游把 q 改成无量纲：上游可算，联动下游 q*q -> ft^2 量纲冲突
    change = _step_change("q", formula="1", output_unit="1")
    from app.errors import ConversionError

    with pytest.raises(ConversionError) as excinfo:
        ra.revise(up_id, 1, change, ca)
    assert excinfo.value.code == "dimension_mismatch"

    # 两个进程视角下两单都保持原样
    for store in (ra, rb):
        u = store.snapshot(up_id)
        d = store.snapshot(down_id)
        assert u["revision"] == 1
        assert d["revision"] == 1
        assert u["results"][1]["value"] == "10"
        assert [r["value"] for r in d["results"]] == ["100", "100"]
        assert d["binding"]["upstream_revision"] == 1
    # 失败发生在落盘之前：磁盘文件里两单修订号也都不变
    raw = json.load(
        open(os.path.join(peers["dir"], revisions.STORE_FILENAME), encoding="utf-8")
    )
    assert raw["sheets"][up_id]["revision"] == 1
    assert raw["sheets"][down_id]["revision"] == 1
    # 仍可用原修订号重试成功
    rec = ra.revise(up_id, 1, VariableChange("x", "20", "ft"), ca)
    assert rec["revision"] == 2
    assert rb.snapshot(down_id)["revision"] == 2


def _step_change(name, *, formula, output_unit):
    from app.revisions import StepChange
    from app.schemas import WorksheetStep

    return StepChange(
        WorksheetStep.model_validate(
            {"name": name, "formula": formula, "output_unit": output_unit}
        )
    )


# ---------------------------------------------------------------------------
# 存档格式与锁文件
# ---------------------------------------------------------------------------


def test_json_archive_format_unchanged_and_lock_file_is_additive(peers):
    a, _ = peers["sets"]
    ra = peers["rev"][0]
    ca = peers["recalc"][0]
    data_dir = peers["dir"]
    set_id = a.publish(_item("ft", "0.3048", "m"))
    body = upstream_body(set_id)
    ra.create(body, set_id, ca(body))

    sets_path = os.path.join(data_dir, unit_sets.STORE_FILENAME)
    sheets_path = os.path.join(data_dir, revisions.STORE_FILENAME)
    sets_raw = json.load(open(sets_path, encoding="utf-8"))
    sheets_raw = json.load(open(sheets_path, encoding="utf-8"))
    # 与单进程时代完全相同的 JSON 结构
    assert sets_raw["version"] == 1 and set_id in sets_raw["sets"]
    assert "aliases" in sets_raw
    assert sheets_raw["version"] == 1
    (sheet,) = sheets_raw["sheets"].values()
    assert sheet["revision"] == 1 and sheet["binding"] is None
    assert set(sheet) == {"revision", "custom_unit_set_id", "worksheet",
                         "results", "binding"}

    listed = set(os.listdir(data_dir))
    assert unit_sets.STORE_FILENAME in listed
    assert revisions.STORE_FILENAME in listed
    # 跨进程锁是 JSON 之外的附加零字节文件
    lock_files = {n for n in listed if n.endswith(".lock")}
    assert lock_files == {
        f"{unit_sets.STORE_FILENAME}.lock",
        f"{revisions.STORE_FILENAME}.lock",
    }
    for name in lock_files:
        assert os.path.getsize(os.path.join(data_dir, name)) == 0


def test_unbound_worksheet_behavior_unchanged_across_processes(peers):
    a, b = peers["sets"]
    ra, rb = peers["rev"]
    ca, cb = peers["recalc"]
    set_id = a.publish(_item("ft", "0.3048", "m"))
    body = upstream_body(set_id)
    wid, created = ra.create(body, set_id, ca(body))
    assert created.get("binding") is None
    # 无关联工作单在另一进程上反复修订均自洽，且不带绑定信息
    for i, val in enumerate(("3", "4", "5"), start=2):
        rec = rb.revise(wid, i - 1, VariableChange("t", val, "h"), cb)
        assert rec["revision"] == i
        assert rec["binding"] is None
        assert ra.snapshot(wid)["revision"] == i
    # 全新“第三个进程”按容器重建方式加载，数据一致
    fresh = revisions.RevisionStore(peers["dir"])
    fresh.load()
    snap = fresh.snapshot(wid)
    assert snap["revision"] == 4
    assert snap["worksheet"]["variables"]["t"]["value"] == "5"
    assert snap["binding"] is None
