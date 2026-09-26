"""引物复测 4×6 孔板排盘。

业务规则
========
从一个历史筛选任务中选取 2–6 条引物做复测，每条引物设 2 或 3 个复孔；
可在 4 行（A–D）× 6 列（1–6）= 24 孔板上预留任意孔位给对照。后端把复孔
分配到剩余空孔，并强制：

- 同一引物的任意两个复孔 **不在同一行、也不在同一列**（类似在「行 × 列」
  二分图上为每条引物选一个匹配）；
- 不同引物的孔位互不重叠；
- 对照孔与已占用孔不参与分配。

搜索方式：**完整回溯（非逐孔贪心）**——按引物选择顺序逐条放置，每条引物
的孔位组合按 **行优先（row-major）** 顺序枚举，第一个成功的完整布局即
返回，因此相同输入的结果稳定且是「引物顺序 × 孔位行优先」字典序下的首个
可行布局。穷举后仍无可行布局时返回 ``feasible=False``，绝不输出部分方案。

纯标准库实现。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 常量与限制
# ---------------------------------------------------------------------------

PLATE_ROWS = 4
PLATE_COLS = 6
PLATE_ROW_LABELS = ("A", "B", "C", "D")
PLATE_WELL_COUNT = PLATE_ROWS * PLATE_COLS

RETEST_MIN_PRIMERS = 2
RETEST_MAX_PRIMERS = 6
RETEST_MIN_REPLICATES = 2
RETEST_MAX_REPLICATES = 3


class PlateError(ValueError):
    """排盘请求的语义错误（重复引物 / 无效孔位 / 容量不足等）。"""


# ---------------------------------------------------------------------------
# 孔位工具
# ---------------------------------------------------------------------------


def parse_well(label: str) -> tuple[int, int]:
    """把 ``"B3"`` 之类的孔位标签解析为 ``(row_index, col_index)``（均 0-based）。

    大小写不敏感（``a1`` 等同 ``A1``）。非法标签抛 :class:`PlateError`。
    """

    if not isinstance(label, str):
        raise PlateError(f"无效孔位 {label!r}：孔位标签需为字符串，如 A1、D6")
    text = label.strip().upper()
    if len(text) < 2 or text[0] not in PLATE_ROW_LABELS:
        raise PlateError(
            f"无效孔位 {label!r}：行标签需为 {'–'.join(PLATE_ROW_LABELS)} 之一，"
            f"如 A1、{PLATE_ROW_LABELS[-1]}{PLATE_COLS}"
        )
    col_text = text[1:]
    if not col_text.isdigit():
        raise PlateError(f"无效孔位 {label!r}：列号需为 1–{PLATE_COLS} 的数字")
    col = int(col_text)
    if not 1 <= col <= PLATE_COLS:
        raise PlateError(
            f"无效孔位 {label!r}：列号需在 1–{PLATE_COLS} 之间（{PLATE_ROWS}×{PLATE_COLS} 孔板）"
        )
    return PLATE_ROW_LABELS.index(text[0]), col - 1


def well_label(row: int, col: int) -> str:
    return f"{PLATE_ROW_LABELS[row]}{col + 1}"


# ---------------------------------------------------------------------------
# 完整搜索
# ---------------------------------------------------------------------------


def _row_masks(n_rows: int, n_cols: int) -> list[int]:
    masks = []
    for r in range(n_rows):
        m = 0
        for c in range(n_cols):
            m |= 1 << (r * n_cols + c)
        masks.append(m)
    return masks


def _col_masks(n_rows: int, n_cols: int) -> list[int]:
    masks = []
    for c in range(n_cols):
        m = 0
        for r in range(n_rows):
            m |= 1 << (r * n_cols + c)
        masks.append(m)
    return masks


def _greedy_matching(cell_mask: int, n_rows: int, n_cols: int, row_bits: list[int]) -> int:
    """贪心匹配（按行扫描、各取第一个未占用列）。结果是最大匹配的下界。"""

    used_cols = 0
    size = 0
    for r in range(n_rows):
        bits = cell_mask & row_bits[r]
        while bits:
            b = bits & -bits
            c = (b.bit_length() - 1) % n_cols
            bits ^= b
            if not (used_cols & (1 << c)):
                used_cols |= 1 << c
                size += 1
                break
    return size


def _max_matching_size(cell_mask: int, n_rows: int, n_cols: int, row_bits: list[int]) -> int:
    """求「行 × 列」二分图（cell_mask 置位即一条边）的最大匹配（Kuhn 算法）。

    最大匹配数是「剩余空孔还能容纳多少个两两异行异列的孔」的上界，用于回溯
    剪枝。板仅 4×6，开销可以忽略（调用方通常先用廉价的贪心匹配试下界）。
    """

    match_col = [-1] * n_cols

    def augment(r: int, seen: list[bool]) -> bool:
        bits = cell_mask & row_bits[r]
        while bits:
            b = bits & -bits
            c = (b.bit_length() - 1) % n_cols
            bits ^= b
            if seen[c]:
                continue
            seen[c] = True
            if match_col[c] == -1 or augment(match_col[c], seen):
                match_col[c] = r
                return True
        return False

    size = 0
    for r in range(n_rows):
        if augment(r, [False] * n_cols):
            size += 1
    return size


def _bits(mask: int):
    """迭代置位比特的索引（低位到高位）。"""

    while mask:
        b = mask & -mask
        yield b.bit_length() - 1
        mask ^= b


def _row_major_placements(n_rows: int, n_cols: int, size: int) -> list[tuple[int, list[int], int, int]]:
    """预生成一块空板上所有取 ``size`` 个互异行、互异列孔位的组合。

    结果严格按 **行优先字典序**（每个组合先按孔位索引排序，组合之间字典序）
    排列，每项为 ``(孔位掩码, 孔位索引列表, 行掩码, 列掩码)``。规模上限：
    3 复孔在 4×6 板上为 480 项（2 复孔 180 项），可在每次搜索内直接顺序枚举。
    结果按板尺寸缓存：同一尺寸只生成一次，跨请求复用。
    """

    cached = _PLACEMENT_CACHE.get((n_rows, n_cols, size))
    if cached is not None:
        return cached

    placements: list[tuple[int, list[int], int, int]] = []

    def build(r: int, picked: list[int], rows_used: int, cols_used: int) -> None:
        if len(picked) == size:
            mask = 0
            for idx in picked:
                mask |= 1 << idx
            placements.append((mask, list(picked), rows_used, cols_used))
            return
        rows_left = n_rows - r
        if rows_left < size - len(picked):
            return
        # 分支 1：第 r 行取一个孔，列号从小到大；分支 2：跳过该行。
        # 该顺序与「行优先孔位序列上取 size 元子集」字典序一致。
        for c in range(n_cols):
            if cols_used & (1 << c):
                continue
            idx = r * n_cols + c
            build(r + 1, picked + [idx], rows_used | (1 << r), cols_used | (1 << c))
        if rows_left - 1 >= size - len(picked):
            build(r + 1, picked, rows_used, cols_used)

    build(0, [], 0, 0)
    placements.sort(key=lambda p: p[1])
    _PLACEMENT_CACHE[(n_rows, n_cols, size)] = placements
    return placements


_PLACEMENT_CACHE: dict[tuple[int, int, int], list] = {}
# 进程启动即预生成标准 4×6 板的 2 / 3 复孔放法，请求路径零生成开销。
for _size in (RETEST_MIN_REPLICATES, RETEST_MAX_REPLICATES):
    _row_major_placements(PLATE_ROWS, PLATE_COLS, _size)


def _search_layout(
    demands: list[int],
    blocked: int,
    n_rows: int = PLATE_ROWS,
    n_cols: int = PLATE_COLS,
) -> tuple[list[list[int]] | None, int]:
    """完整回溯求首个可行布局。

    参数
    ----
    demands:
        每条引物（按选择顺序）的复孔数。
    blocked:
        对照孔位掩码（行优先：孔 (r,c) 对应位 ``r*n_cols+c``）。

    返回 ``(assignments, nodes)``：可行时 ``assignments[i]`` 为第 i 条引物的
    孔位索引列表（行优先顺序，与复孔序号一致）；不可行为 ``(None, nodes)``。

    实现
    ----
    两层搜索，保证结果是「引物选择顺序 × 孔位行优先」字典序下的 **首个**
    可行布局，同时不是逐孔贪心：

    - 外层 :meth:`lex_dfs` 严格按用户的引物顺序逐条放置，每条引物的候选放法
      按行优先枚举；对每个候选调用可行性判定，**第一个尾段仍可行的候选**
      才被提交（保证字典序最小）。
    - 内层 :meth:`feas` 是带记忆化的完整穷举（自由选择引物顺序：复孔最多的
      引物先放，尽早撞墙），只回答「引物集合 mask 在已占孔位 occupied 下能否
      全部放完」。它仍是 **完整** 穷举，不会把可行候选误判为不可行，从而让
      外层在确定无解的候选上立即跳过、在真正首个可行候选上继续。
    """

    total_cells = n_rows * n_cols
    all_mask = (1 << total_cells) - 1
    free_mask = all_mask & ~blocked
    row_bits = _row_masks(n_rows, n_cols)
    col_bits = _col_masks(n_rows, n_cols)

    # 本板上 2 / 3 复孔的全部合法放法（行优先字典序），搜索时直接顺序枚举。
    placements_by_need = {
        need: _row_major_placements(n_rows, n_cols, need)
        for need in sorted(set(demands))
    }

    n = len(demands)
    full_mask = (1 << n) - 1
    nodes = 0

    def capacity_ok(open_mask: int, mask: int) -> bool:
        """容量必要条件（安全上界剪枝；全部为「必须满足否则必无解」）。"""

        k = mask.bit_count()
        need_total = sum(demands[i] for i in _bits(mask))
        # 1) 空孔总数容不下全部复孔。
        if need_total > open_mask.bit_count():
            return False
        # 2) 每行 / 每列对每条引物至多贡献 1 个孔，故可容纳复孔总数不超过
        #    Σ行 min(该行空孔数, 引物数)（列同理）。
        row_cap = 0
        for r in range(n_rows):
            row_cap += min((open_mask & row_bits[r]).bit_count(), k)
        if row_cap < need_total:
            return False
        col_cap = 0
        for c in range(n_cols):
            col_cap += min((open_mask & col_bits[c]).bit_count(), k)
        if col_cap < need_total:
            return False
        # 3) 复孔最多的那条引物：剩余空孔二分图必须有不小于其复孔数的匹配
        #    （它的各复孔要互异行且互异列）。贪心先试，不足再跑精确增广。
        max_demand = max(demands[i] for i in _bits(mask))
        if max_demand > 2 and _greedy_matching(open_mask, n_rows, n_cols, row_bits) < max_demand:
            if _max_matching_size(open_mask, n_rows, n_cols, row_bits) < max_demand:
                return False
        return True

    # -- 内层：给定引物集合 / 已占孔位的可行性判定（记忆化完整穷举）----------
    memo: dict[tuple[int, int], bool] = {}

    def feas(mask: int, occupied: int) -> bool:
        nonlocal nodes
        if mask == 0:
            return True
        key = (mask, occupied)
        cached = memo.get(key)
        if cached is not None:
            return cached
        nodes += 1

        open_mask = free_mask & ~occupied
        if not capacity_ok(open_mask, mask):
            memo[key] = False
            return False

        # 复孔最多的引物优先（其次编号小者），最快撞墙。
        pi = max(_bits(mask), key=lambda i: (demands[i], -i))
        rest = mask ^ (1 << pi)
        for place_mask, _cells, _r, _c in placements_by_need[demands[pi]]:
            if place_mask & ~open_mask:
                continue
            if feas(rest, occupied | place_mask):
                memo[key] = True
                return True
        memo[key] = False
        return False

    # -- 外层：引物顺序 × 行优先，提交首个尾段可行的候选 --------------------
    answer: list[list[int] | None] = [None] * n

    def lex_dfs(pi: int, occupied: int) -> bool:
        if pi == n:
            return True
        open_mask = free_mask & ~occupied
        # 尚未放置的引物 pi+1..n-1（低 pi+1 位对应引物 0..pi，需排除）。
        tail = full_mask ^ ((1 << (pi + 1)) - 1)
        for place_mask, cells, _r, _c in placements_by_need[demands[pi]]:
            if place_mask & ~open_mask:
                continue
            new_occ = occupied | place_mask
            # 尾段完整可行性判定：可行才提交该候选（保证取到首个可行布局）。
            if feas(tail, new_occ):
                answer[pi] = cells
                if lex_dfs(pi + 1, new_occ):
                    return True
        return False

    if not capacity_ok(free_mask, full_mask):
        return None, nodes
    if not lex_dfs(0, 0):
        return None, nodes
    return [list(cells) for cells in answer], nodes  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 顶层排盘流程
# ---------------------------------------------------------------------------


def retest_layout(
    known_primers: list[str],
    items: list[dict],
    control_wells: list[str] | None = None,
) -> dict:
    """对历史任务选中的引物执行复测排盘。

    参数
    ----
    known_primers:
        该历史任务保存的全部引物 id（按任务中的顺序）。
    items:
        ``[{"primer_id": ..., "replicates": 2|3}, ...]``，顺序即引物选择顺序。
    control_wells:
        预留给对照的孔位标签（如 ``["A1", "H7-不存在"]``），可省略。

    返回可直接 JSON 序列化的布局或无解诊断；请求本身可修正的错误
    （重复引物、无效孔位、容量不足等）抛 :class:`PlateError`。
    """

    control_wells = control_wells or []
    known_set = set(known_primers)

    # -- 引物与复孔数校验 ----------------------------------------------------
    if not RETEST_MIN_PRIMERS <= len(items) <= RETEST_MAX_PRIMERS:
        raise PlateError(
            f"复测需选择 {RETEST_MIN_PRIMERS}–{RETEST_MAX_PRIMERS} 条引物，"
            f"当前为 {len(items)} 条"
        )

    selected: list[str] = []
    demands: list[int] = []
    seen: set[str] = set()
    for item in items:
        pid = item.get("primer_id")
        reps = item.get("replicates")
        if not isinstance(pid, str) or not pid:
            raise PlateError("每条引物需提供字符串类型的 primer_id")
        if pid in seen:
            raise PlateError(f"引物 {pid} 被重复选择，同一条引物只能出现一次")
        if pid not in known_set:
            raise PlateError(f"引物 {pid} 不属于该任务，请只选择本任务中的引物")
        if not isinstance(reps, int) or isinstance(reps, bool):
            raise PlateError(f"引物 {pid} 的复孔数需为整数（2 或 3）")
        if not RETEST_MIN_REPLICATES <= reps <= RETEST_MAX_REPLICATES:
            raise PlateError(
                f"引物 {pid} 的复孔数需为 {RETEST_MIN_REPLICATES} 或 "
                f"{RETEST_MAX_REPLICATES}，当前为 {reps}"
            )
        seen.add(pid)
        selected.append(pid)
        demands.append(reps)

    # -- 对照孔位校验 --------------------------------------------------------
    if len(control_wells) > PLATE_WELL_COUNT:
        raise PlateError(f"对照孔最多 {PLATE_WELL_COUNT} 个，当前为 {len(control_wells)} 个")
    controls: list[tuple[int, int]] = []
    seen_wells: set[str] = set()
    for raw in control_wells:
        r, c = parse_well(raw)
        label = well_label(r, c)
        if label in seen_wells:
            raise PlateError(f"对照孔 {label} 被重复预留")
        seen_wells.add(label)
        controls.append((r, c))
    controls.sort(key=lambda rc: (rc[0], rc[1]))

    blocked = 0
    for r, c in controls:
        blocked |= 1 << (r * PLATE_COLS + c)

    total_replicates = sum(demands)
    available = PLATE_WELL_COUNT - len(controls)

    base = {
        "plate": {"rows": PLATE_ROWS, "cols": PLATE_COLS,
                  "row_labels": list(PLATE_ROW_LABELS)},
        "primer_ids": selected,
        "primer_count": len(selected),
        "items": [
            {"primer_id": pid, "replicates": reps}
            for pid, reps in zip(selected, demands)
        ],
        "control_wells": [well_label(r, c) for r, c in controls],
        "control_count": len(controls),
        "replicate_total": total_replicates,
        "available_wells": available,
        "empty_well_count": available - total_replicates,
    }

    # -- 容量校验（总量不足属于可修正的请求错误，422）------------------------
    if total_replicates > available:
        raise PlateError(
            f"孔板容量不足：{len(selected)} 条引物共需 {total_replicates} 个复孔，"
            f"预留对照 {len(controls)} 孔后仅剩 {available} 个可用孔"
            f"（{PLATE_ROWS}×{PLATE_COLS}={PLATE_WELL_COUNT} 孔）。"
            "请减少引物数量、降低复孔数或减少对照预留孔。"
        )

    # -- 完整搜索 ------------------------------------------------------------
    assignments, nodes = _search_layout(demands, blocked)

    if assignments is None:
        open_mask = ((1 << PLATE_WELL_COUNT) - 1) & ~blocked
        free_wells = [
            well_label(idx // PLATE_COLS, idx % PLATE_COLS)
            for idx in range(PLATE_WELL_COUNT)
            if (open_mask >> idx) & 1
        ]
        row_free = [
            {"row": PLATE_ROW_LABELS[r],
             "free": (open_mask & _row_masks(PLATE_ROWS, PLATE_COLS)[r]).bit_count()}
            for r in range(PLATE_ROWS)
        ]
        col_bits = _col_masks(PLATE_ROWS, PLATE_COLS)
        col_free = [
            {"col": c + 1, "free": (open_mask & col_bits[c]).bit_count()}
            for c in range(PLATE_COLS)
        ]
        return {
            **base,
            "feasible": False,
            "reason": "no_valid_layout",
            "message": (
                f"已对 {len(selected)} 条引物（共 {total_replicates} 个复孔）穷尽搜索全部"
                f"排法（访问 {nodes} 个搜索状态），在当前 {len(controls)} 个对照孔"
                f"（{'、'.join(base['control_wells']) or '无'}）的约束下，"
                "无法让每条引物的复孔都分属不同的行与列。"
                "请减少或调整对照预留孔位（避免整行/整列被占）、减少引物数量，"
                "或将部分引物的 3 复孔改为 2 复孔后重试；后端不会输出部分方案。"
            ),
            "search_nodes": nodes,
            "free_wells": free_wells,
            "free_per_row": row_free,
            "free_per_col": col_free,
        }

    # -- 构造布局：孔板网格 + 每引物孔位清单 ---------------------------------
    cell_owner: dict[int, tuple[int, int]] = {}
    items_out: list[dict] = []
    for pi, (pid, cells) in enumerate(zip(selected, assignments)):
        wells = []
        for rep_idx, idx in enumerate(cells, start=1):
            r, c = divmod(idx, PLATE_COLS)
            cell_owner[idx] = (pi, rep_idx)
            wells.append(
                {"well": well_label(r, c), "row": PLATE_ROW_LABELS[r],
                 "col": c + 1, "replicate_index": rep_idx}
            )
        rows_used = sorted({w["row"] for w in wells})
        cols_used = sorted(w["col"] for w in wells)
        items_out.append(
            {
                "primer_id": pid,
                "replicates": demands[pi],
                "wells": wells,
                "rows": rows_used,
                "cols": cols_used,
            }
        )

    control_set = set(controls)
    grid = []
    for r in range(PLATE_ROWS):
        row_cells = []
        for c in range(PLATE_COLS):
            idx = r * PLATE_COLS + c
            cell = {"row": PLATE_ROW_LABELS[r], "col": c + 1,
                    "well": well_label(r, c)}
            if (r, c) in control_set:
                cell.update({"status": "control", "item_index": None,
                             "primer_id": None, "replicate_index": None})
            elif idx in cell_owner:
                pi, rep_idx = cell_owner[idx]
                cell.update({"status": "primer", "item_index": pi,
                             "primer_id": selected[pi], "replicate_index": rep_idx})
            else:
                cell.update({"status": "empty", "item_index": None,
                             "primer_id": None, "replicate_index": None})
            row_cells.append(cell)
        grid.append(row_cells)

    return {
        **base,
        "feasible": True,
        "reason": None,
        "message": (
            f"完整搜索找到可行布局：{len(selected)} 条引物、{total_replicates} 个复孔"
            f"已分配到 {PLATE_ROWS}×{PLATE_COLS} 孔板"
            f"（对照 {len(controls)} 孔，余空 {available - total_replicates} 孔）。"
        ),
        "search_nodes": nodes,
        "grid": grid,
        "items": items_out,
    }
