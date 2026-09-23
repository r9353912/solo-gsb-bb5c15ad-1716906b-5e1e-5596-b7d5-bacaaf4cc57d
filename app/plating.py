"""引物复测排盘（4×6 孔板，复孔不同行不同列的完整搜索）。

问题描述
========
从一个历史筛选任务中选出 ``2–6`` 条引物，每条引物在 **4 行 × 6 列**（共 24
孔，行号 A–D、列号 1–6）的孔板上占 ``2`` 或 ``3`` 个复孔。用户可把任意孔位
预留为对照孔，这些孔不参与分配。约束：

* 每个孔至多放置一个复孔；
* 同一条引物的复孔两两 **不在同一行、也不在同一列**；
* 不同引物之间没有行/列约束（只要求孔位不重叠）。

搜索方式
========
**完整（穷举）回溯搜索**，而不是逐孔贪心：按引物选择顺序逐条放置，每条引物
的候选为「剩余空孔中满足不同行不同列的孔位组合」，候选组合按 **行优先顺序**
（A1 < A2 < … < D6）排序，故找到的第一个解就是题目要求的「按引物选择顺序与
孔位行优先顺序的首个有效布局」，且对同一输入稳定可复现。

剪枝：
  * 容量不足 / 开放行列数不足在搜索前直接判定；
  * 每步放置后调用一个 **精确可行性预言** 检查剩余空孔：以行优先最小的剩余孔
    为锚点（属于下一组复孔 / 作为闲置孔）做记忆化递归，必要条件含孔数下界、
    行/列容量界（每行至多被每条引物占 1 孔）与「孔-行-列」二分匹配，不可行
    分支立即剪去，使无可行布局时也能快速穷尽。

无可行布局时只返回 ``feasible: False`` 与可读原因，**绝不输出部分方案**。
"""

from __future__ import annotations

import re
from itertools import combinations

PLATE_ROWS = 4  # A–D
PLATE_COLS = 6  # 1–6
PLATE_WELLS = PLATE_ROWS * PLATE_COLS

RETEST_MIN_PRIMERS = 2
RETEST_MAX_PRIMERS = 6
RETEST_REPLICATE_CHOICES = (2, 3)

_ROW_LETTERS = "ABCD"
_WELL_RE = re.compile(r"^([A-Da-d])([1-6])$")


class PlatingError(ValueError):
    """排盘请求的语义错误（引物 / 复孔数 / 孔位不合法或容量不足）。"""


def _well_index(well: str) -> int:
    """把 ``A1``–``D6`` 形式的孔位解析为行优先线性下标 0..23（非法则报错）。"""

    m = _WELL_RE.match(well.strip())
    if not m:
        raise PlatingError(
            f"孔位 {well!r} 无效：需为行字母 A–D 与列号 1–6 的组合（如 A1、C6）"
        )
    return _ROW_LETTERS.index(m.group(1).upper()) * PLATE_COLS + int(m.group(2)) - 1


def _well_label(idx: int) -> str:
    """行优先线性下标 -> ``A1`` 形式孔位。"""

    return f"{_ROW_LETTERS[idx // PLATE_COLS]}{idx % PLATE_COLS + 1}"


def normalize_wells(wells: list[str]) -> list[str]:
    """校验并规范化孔位列表：大写、合法、不重复，按行优先顺序返回。"""

    labels: list[str] = []
    seen: set[int] = set()
    for w in wells:
        if not isinstance(w, str):
            raise PlatingError(f"孔位必须为字符串，收到 {w!r}")
        idx = _well_index(w)
        if idx in seen:
            raise PlatingError(f"预留对照孔位重复指定：{_well_label(idx)}")
        seen.add(idx)
        labels.append(_well_label(idx))
    labels.sort(key=lambda w: _well_index(w))
    return labels


def _max_pairwise_distinct_pairs(open_mask: int, k: int) -> int:
    """剩余空孔中最多能选出多少个「两两不同行不同列」的孔。

    这正是「孔 -> 行 / 列」二分图的最大匹配：每个孔是一条连接其所在行与列的边，
    匹配数即两两不同行不同列的最大孔数。k ≤ 3，用 Kuhn 增广路即可。
    """

    adj: list[list[int]] = [[] for _ in range(PLATE_ROWS)]
    for idx in range(PLATE_WELLS):
        if open_mask & (1 << idx):
            adj[idx // PLATE_COLS].append(idx % PLATE_COLS)

    match_col = [-1] * PLATE_COLS

    def augment(r: int, seen_cols: list[bool]) -> bool:
        for c in adj[r]:
            if seen_cols[c]:
                continue
            seen_cols[c] = True
            if match_col[c] == -1 or augment(match_col[c], seen_cols):
                match_col[c] = r
                return True
        return False

    matched = 0
    for r in range(PLATE_ROWS):
        if augment(r, [False] * PLATE_COLS):
            matched += 1
            if matched >= k:
                return matched
    return matched


def retest_plate_layout(
    primers: list[tuple[str, str]],
    selected_ids: list[str],
    replicates: int,
    reserved_wells: list[str] | None = None,
) -> dict:
    """为复测排盘执行完整搜索，返回可直接序列化的布局或无解诊断。

    ``selected_ids`` 的顺序即引物选择顺序；返回方案中引物也按该顺序排列。
    """

    # -- 输入语义校验 -------------------------------------------------------
    known = [pid for pid, _ in primers]
    known_set = set(known)

    if not RETEST_MIN_PRIMERS <= len(selected_ids) <= RETEST_MAX_PRIMERS:
        raise PlatingError(
            f"复测排盘需要选择 {RETEST_MIN_PRIMERS}–{RETEST_MAX_PRIMERS} 条引物，"
            f"当前为 {len(selected_ids)} 条"
        )
    if len(set(selected_ids)) != len(selected_ids):
        dup = sorted({pid for pid in selected_ids if selected_ids.count(pid) > 1})
        raise PlatingError(f"选中引物存在重复，请去重后再提交：{'、'.join(dup)}")
    unknown = [pid for pid in selected_ids if pid not in known_set]
    if unknown:
        raise PlatingError(f"以下引物不属于该任务：{'、'.join(unknown)}")
    if replicates not in RETEST_REPLICATE_CHOICES:
        raise PlatingError(
            f"每条引物的复孔数只能为 {RETEST_REPLICATE_CHOICES[0]} 或 "
            f"{RETEST_REPLICATE_CHOICES[1]}，当前为 {replicates}"
        )

    reserved = normalize_wells(list(reserved_wells or []))
    reserved_mask = 0
    for w in reserved:
        reserved_mask |= 1 << _well_index(w)

    primer_count = len(selected_ids)
    needed = primer_count * replicates
    open_mask = ((1 << PLATE_WELLS) - 1) & ~reserved_mask
    open_wells = [i for i in range(PLATE_WELLS) if open_mask & (1 << i)]
    open_count = len(open_wells)

    base = {
        "rows": PLATE_ROWS,
        "columns": PLATE_COLS,
        "primer_ids": list(selected_ids),
        "primer_count": primer_count,
        "replicates": replicates,
        "total_wells": PLATE_WELLS,
        "reserved_wells": reserved,
        "reserved_count": len(reserved),
        "open_wells": [_well_label(i) for i in open_wells],
        "open_count": open_count,
        "wells_needed": needed,
    }

    def grid(assignment: dict[int, int] | None) -> list[list[dict]]:
        assignment = assignment or {}
        grid_rows: list[list[dict]] = []
        for r in range(PLATE_ROWS):
            row: list[dict] = []
            for c in range(PLATE_COLS):
                idx = r * PLATE_COLS + c
                if reserved_mask & (1 << idx):
                    cell = {"row": _ROW_LETTERS[r], "column": c + 1, "well": _well_label(idx),
                            "status": "control"}
                elif idx in assignment:
                    pi, rep = assignment[idx]
                    cell = {"row": _ROW_LETTERS[r], "column": c + 1, "well": _well_label(idx),
                            "status": "primer", "primer_index": pi + 1,
                            "primer_id": selected_ids[pi], "replicate_no": rep}
                else:
                    cell = {"row": _ROW_LETTERS[r], "column": c + 1, "well": _well_label(idx),
                            "status": "empty"}
                row.append(cell)
            grid_rows.append(row)
        return grid_rows

    # -- 容量不足：请求可修正的语义错误 -> 422 ------------------------------
    if needed > open_count:
        raise PlatingError(
            f"容量不足：{primer_count} 条引物 × {replicates} 个复孔共需 {needed} 个空孔，"
            f"扣除 {len(reserved)} 个对照预留孔后仅剩 {open_count} 个空孔。"
            "请减少引物或复孔数，或释放部分预留孔后重试。"
        )

    # 不同行不同列至少需要 k 个开放行与 k 个开放列。
    open_row_count = sum(
        1 for r in range(PLATE_ROWS)
        if any(open_mask & (1 << (r * PLATE_COLS + c)) for c in range(PLATE_COLS))
    )
    open_col_count = sum(
        1 for c in range(PLATE_COLS)
        if any(open_mask & (1 << (r * PLATE_COLS + c)) for r in range(PLATE_ROWS))
    )

    def infeasible(reason: str, message: str) -> dict:
        return {
            **base,
            "feasible": False,
            "reason": reason,
            "message": message,
            "grid": grid(None),
            "assignments": [],
        }

    if open_row_count < replicates or open_col_count < replicates:
        return infeasible(
            "insufficient_open_rows_or_columns",
            f"每条引物的 {replicates} 个复孔需落在不同行、不同列，因此至少需要 "
            f"{replicates} 个未被对照完全占满的行与列；当前开放行 {open_row_count} 个、"
            f"开放列 {open_col_count} 个，无可行布局。请释放部分预留孔后重试。"
            "（不会输出任何部分方案。）",
        )

    # -- 完整搜索：候选组合按行优先顺序枚举 --------------------------------
    # 每个候选是一个按行优先排好序的 k 元孔位下标组（天然不同行不同列）。
    candidates: list[tuple[int, ...]] = [
        combo
        for combo in combinations(open_wells, replicates)
        if len({i // PLATE_COLS for i in combo}) == replicates
        and len({i % PLATE_COLS for i in combo}) == replicates
    ]

    chosen: list[tuple[int, ...]] = []

    col_bit_mask = sum(1 << (r * PLATE_COLS) for r in range(PLATE_ROWS))

    def _necessary_ok(mask: int, m: int) -> bool:
        """必要条件快速检查（还剩 m 条引物、mask 为剩余空孔）。

        1. 孔数下界与「单引物可取 k 个不同行不同列孔」的二分匹配检查；
        2. 行/列容量界：总余量 s = 空孔数 - m·k 是全局最多可闲置的孔数。
           每行最多被 m 条引物各占 1 孔，故某行开放孔数 d 必须满足
           d ≤ m + s（即使把全部 s 个闲置孔都算在该行，仍至少要占 d-s 孔）；
           列同理。
        """

        open_n = mask.bit_count()
        if open_n < m * replicates:
            return False
        if _max_pairwise_distinct_pairs(mask, replicates) < replicates:
            return False
        slack = open_n - m * replicates
        for r in range(PLATE_ROWS):
            row_bits = mask >> (r * PLATE_COLS) & ((1 << PLATE_COLS) - 1)
            if row_bits.bit_count() > m + slack:
                return False
        for c in range(PLATE_COLS):
            if ((mask >> c) & col_bit_mask).bit_count() > m + slack:
                return False
        return True

    # 每个孔对应的「包含该孔的全部有效候选」（位掩码），供精确可行性
    # 预言枚举锚点组合使用。
    combos_through: list[list[int]] = [[] for _ in range(PLATE_WELLS)]
    for combo in candidates:
        cmask = 0
        for idx in combo:
            cmask |= 1 << idx
        for idx in combo:
            combos_through[idx].append(cmask)

    def can_pack(mask: int, m: int) -> bool:
        """精确可行性预言：mask 的空孔中能否放下 m 条引物的复孔组合。

        引物在此不可区分（只问子树是否可行），用锚点法消除引物排列对称性：
        取行优先顺序最小的开放孔——任何完整方案中它要么属于某条引物的一组
        复孔，要么是闲置孔（仅当余量 s > 0）。两种情况递归即可；状态
        (mask, m) 记忆化。规模极小（4×6、m ≤ 6），可快速穷尽。
        """

        memo: dict[tuple[int, int], bool] = {}

        def rec(cmask: int, cm: int) -> bool:
            key = (cmask, cm)
            cached = memo.get(key)
            if cached is not None:
                return cached
            if cm == 0:
                result = True
            elif not _necessary_ok(cmask, cm):
                result = False
            else:
                anchor = (cmask & -cmask).bit_length() - 1
                result = False
                # 情况 1：锚点孔属于下一条引物的某组复孔。
                for option in combos_through[anchor]:
                    if option & ~cmask:
                        continue
                    if rec(cmask & ~option, cm - 1):
                        result = True
                        break
                # 情况 2：锚点孔闲置（至少要剩一个闲置名额）。
                if not result and cmask.bit_count() > cm * replicates:
                    result = rec(cmask & ~(1 << anchor), cm)
            memo[key] = result
            return result

        return rec(mask, m)

    def dfs(pi: int, used_mask: int) -> bool:
        if pi == primer_count:
            return True
        remaining = primer_count - pi - 1
        for combo in candidates:
            combo_mask = 0
            for idx in combo:
                combo_mask |= 1 << idx
            if combo_mask & used_mask:
                continue
            # 精确前向检查：扣除本组合后必须仍能容纳其余引物（不可行子树
            # 立即剪去，保证无可行布局时也能快速穷尽）。
            future_open = open_mask & ~(used_mask | combo_mask)
            if remaining > 0 and not can_pack(future_open, remaining):
                continue
            chosen.append(combo)
            if dfs(pi + 1, used_mask | combo_mask):
                return True
            chosen.pop()
        return False

    if not dfs(0, 0):
        return infeasible(
            "no_valid_layout",
            f"完整搜索已穷举全部可能布局：在扣除 {len(reserved)} 个对照预留孔后，"
            f"无法为 {primer_count} 条引物各放置 {replicates} 个不同行不同列的复孔"
            "（不同引物间也不能共用孔位），无可行布局。"
            "请释放部分预留孔、减少引物或改为 2 个复孔后重试。"
            "（不会输出任何部分方案。）",
        )

    # -- 构造稳定输出：复孔按孔位行优先顺序编号 1..k ------------------------
    assignment: dict[int, tuple[int, int]] = {}
    assignments_out: list[dict] = []
    for pi, combo in enumerate(chosen):
        wells = [_well_label(idx) for idx in combo]
        for rep, idx in enumerate(combo, start=1):
            assignment[idx] = (pi, rep)
        assignments_out.append(
            {
                "primer_index": pi + 1,
                "primer_id": selected_ids[pi],
                "replicates": replicates,
                "wells": wells,
            }
        )

    return {
        **base,
        "feasible": True,
        "reason": None,
        "message": (
            f"已为 {primer_count} 条引物各分配 {replicates} 个复孔（共 {needed} 孔），"
            f"另预留 {len(reserved)} 个对照孔；同一引物的复孔均不同行、不同列。"
            "布局为完整搜索按引物选择顺序与孔位行优先顺序找到的首个有效布局。"
        ),
        "grid": grid(assignment),
        "assignments": assignments_out,
    }
