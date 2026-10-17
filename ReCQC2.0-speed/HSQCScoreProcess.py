from typing import Union


class CarbonCoordinateExtractor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """创建坐标存储表"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DROP TABLE IF EXISTS carbon_h_coordinates")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS carbon_h_coordinates (
                    coord_id INTEGER PRIMARY KEY,
                    mol_id INTEGER,
                    c_index INTEGER,
                    h_count INTEGER,
                    point_count INTEGER,
                    coordinates TEXT,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)
            conn.execute(
                "CREATE INDEX idx_coordinates_mol_type ON carbon_h_coordinates(mol_id, h_count)"
            )
            conn.commit()

    def _process_carbon(self, c_data: dict) -> Union[None, dict]:
        """处理单个碳原子数据"""
        h_count = c_data["h_count"]
        c_shift = c_data["c_shift"]
        h_shifts = json.loads(c_data["h_shifts"])

        # 基础校验
        if h_count == 0 or c_shift is None:
            return None
        if len(h_shifts) != h_count:
            print(
                f"Carbon atom {c_data['c_index']} hydrogen displacement data mismatch, expected {h_count}, actual {len(h_shifts)}"
            )
            return None

        points = []
        try:
            if h_count == 1:
                if h_shifts[0] is not None:
                    points.append((c_shift, h_shifts[0]))

            elif h_count == 2:
                for hs in h_shifts:
                    if hs is not None:
                        points.append((c_shift, hs))

            elif h_count == 3:
                valid_shifts = [round(hs, 2) for hs in h_shifts if hs is not None]
                if valid_shifts:
                    avg = round(sum(valid_shifts) / len(valid_shifts), 2)  # 修正点
                    points.append((round(c_shift, 2), avg))

            else:  # 处理异常情况
                print(
                    f"Unconventional hydrogen number {h_count} @ mol:{c_data['mol_id']}-c:{c_data['c_index']}"
                )
                return None

        except TypeError as e:
            print(f"Data Processing Errors: {str(e)}")
            return None

        return {
            "mol_id": c_data["mol_id"],
            "c_index": c_data["c_index"],
            "h_count": h_count,
            "points": points,
            "count": len(points),
        }

    def process_all(self):
        """批量处理所有分子（修正进度显示）"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row

            # 获取总碳原子数（h_count>0且c_shift存在）
            total_query = """
                SELECT COUNT(*)
                FROM carbons
                WHERE h_count > 0
                  AND c_shift IS NOT NULL
            """
            total = conn.execute(total_query).fetchone()[0]

            # 获取数据游标
            cursor = conn.execute("""
                SELECT c.mol_id, c.c_index, c.h_count, c.c_shift, c.h_shifts
                FROM carbons c
                WHERE c.h_count > 0
                  AND c.c_shift IS NOT NULL
                ORDER BY c.mol_id, c.c_index
            """)

            batch = []
            pending_progress = 0
            with tqdm(total=total, desc="Dealing with carbon atoms", unit="carbon") as pbar:
                for row in cursor:
                    # 处理每个碳原子（无论是否生成坐标点）
                    result = self._process_carbon(dict(row))
                    if result and result["count"] > 0:
                        batch.append(
                            (
                                result["mol_id"],
                                result["c_index"],
                                result["h_count"],
                                result["count"],
                                json.dumps(result["points"]),
                            )
                        )

                    pending_progress += 1

                    # 批量插入
                    if len(batch) >= 500:
                        conn.executemany(
                            """
                            INSERT INTO carbon_h_coordinates
                            (mol_id, c_index, h_count, point_count, coordinates)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            batch,
                        )
                        batch = []
                        pbar.update(pending_progress)
                        pending_progress = 0

                # 插入剩余数据
                if batch:
                    conn.executemany(
                        """
                        INSERT INTO carbon_h_coordinates
                        (mol_id, c_index, h_count, point_count, coordinates)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        batch,
                    )
                    conn.commit()

                if pending_progress:
                    pbar.update(pending_progress)

            # 获取实际插入的记录数
            inserted = conn.execute("SELECT COUNT(*) FROM carbon_h_coordinates").fetchone()[0]
            print(
                f"Processing is complete! Scans {total} carbon atoms to generate {inserted} valid coordinate records"
            )


import json
import sqlite3
from contextlib import closing

import numpy as np
from tqdm import tqdm


def _calculate_hsqc_point_score(db_point, user_point, c_tolerance, h_tolerance):
    if tuple(user_point) == (0, 0):
        return 0.0

    db_c, db_h = db_point
    user_c, user_h = user_point
    if (
        db_c - c_tolerance <= user_c <= db_c + c_tolerance
        and db_h - h_tolerance <= user_h <= db_h + h_tolerance
    ):
        return 1.0

    delta_c = user_c - db_c
    delta_h = user_h - db_h
    if delta_c == 0:
        region = "top" if delta_h > 0 else "bottom"
    elif abs(delta_h / delta_c) > h_tolerance / c_tolerance:
        region = "top" if delta_h > 0 else "bottom"
    else:
        region = "right" if delta_c > 0 else "left"

    if region in ("left", "right"):
        denominator = abs(delta_c)
        score = c_tolerance / denominator if denominator else 0.0
    else:
        denominator = abs(delta_h)
        score = h_tolerance / denominator if denominator else 0.0
    return max(0.0, min(score, 1.0))


class UserPointMatcher:
    def __init__(
        self,
        db_path: str,
        user_points: dict,
        mode: int = 1,
        CH_merge="N",
        c_tolerance: float = 1.0,
        h_tolerance: float = 0.2,
    ):
        self.db_path = db_path
        self.user_points = user_points
        self.mode = mode
        self.CH_merge = CH_merge
        self.c_tol = float(c_tolerance)
        self.h_tol = float(h_tolerance)
        self._validate_input()
        self._create_result_table()

    def _validate_input(self):
        if self.CH_merge not in {"Y", "N"}:
            raise ValueError("CH_merge must be Y or N")
        if self.mode == 1:
            if "All_type" not in self.user_points:
                raise ValueError("Mode 1 requires All_type key")
        elif self.mode == 2:
            if not ("type13" in self.user_points and "type2" in self.user_points):
                raise ValueError("Mode 2 requires the type13 and type2 keys")
        elif self.mode == 3:
            if not all(k in self.user_points for k in ["type1", "type2", "type3"]):
                raise ValueError("Mode 3 requires the type1/type2/type3 keys")
        else:
            raise ValueError("Invalid Mode Selection (1-3)")

        if self.c_tol <= 0 or self.h_tol <= 0:
            raise ValueError("HSQC tolerances must be positive")

    def _create_result_table(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DROP TABLE IF EXISTS user_matches")
            conn.execute("""
                CREATE TABLE user_matches (
                    match_id INTEGER PRIMARY KEY,
                    mol_id INTEGER,
                    c_index INTEGER,
                    db_point TEXT,
                    user_point TEXT,
                    distance REAL,
                    h_type INTEGER,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id))
            """)
            conn.execute("CREATE INDEX idx_match_h ON user_matches(h_type)")
            conn.execute("CREATE INDEX idx_match_mol ON user_matches(mol_id)")
            conn.commit()

    def _calculate_distances(self, db_points, user_points):
        if not db_points or not user_points:
            return []

        db = np.asarray(db_points, dtype=float)
        user = np.asarray(user_points, dtype=float)
        delta_c = db[:, 0, np.newaxis] - user[np.newaxis, :, 0]
        delta_h = (db[:, 1, np.newaxis] - user[np.newaxis, :, 1]) * 10.0
        rounded = np.round(np.sqrt(delta_c * delta_c + delta_h * delta_h), 2).ravel()

        # Row-major flattening has the same i-then-j order as the original
        # nested loops. A stable sort therefore preserves identical tie order.
        order = np.argsort(rounded, kind="stable")
        user_count = len(user_points)
        return [
            (int(flat // user_count), int(flat % user_count), float(rounded[flat]))
            for flat in order
        ]

    @staticmethod
    def _point_distance(db_point, user_point):
        delta_c = db_point[0] - user_point[0]
        delta_h = (db_point[1] - user_point[1]) * 10.0
        return float(np.hypot(delta_c, delta_h))

    @staticmethod
    def _linear_sum_assignment(cost_matrix):
        """Solve a rectangular minimum-cost assignment using NumPy."""
        cost = np.asarray(cost_matrix, dtype=np.float64)
        row_count, column_count = cost.shape
        if row_count == 0:
            return [], []
        if row_count > column_count:
            raise ValueError("The number of rows must not exceed the number of columns")

        row_potential = np.zeros(row_count + 1, dtype=np.float64)
        column_potential = np.zeros(column_count + 1, dtype=np.float64)
        matched_row = np.zeros(column_count + 1, dtype=np.int32)
        predecessor = np.zeros(column_count + 1, dtype=np.int32)

        for row in range(1, row_count + 1):
            matched_row[0] = row
            current_column = 0
            minimum_cost = np.full(column_count + 1, np.inf, dtype=np.float64)
            visited = np.zeros(column_count + 1, dtype=bool)
            while True:
                visited[current_column] = True
                current_row = matched_row[current_column]
                delta = np.inf
                next_column = 0
                for column in range(1, column_count + 1):
                    if visited[column]:
                        continue
                    reduced = (
                        cost[current_row - 1, column - 1]
                        - row_potential[current_row]
                        - column_potential[column]
                    )
                    if reduced < minimum_cost[column]:
                        minimum_cost[column] = reduced
                        predecessor[column] = current_column
                    if minimum_cost[column] < delta:
                        delta = minimum_cost[column]
                        next_column = column
                for column in range(column_count + 1):
                    if visited[column]:
                        row_potential[matched_row[column]] += delta
                        column_potential[column] -= delta
                    else:
                        minimum_cost[column] -= delta
                current_column = next_column
                if matched_row[current_column] == 0:
                    break
            while True:
                previous_column = predecessor[current_column]
                matched_row[current_column] = matched_row[previous_column]
                current_column = previous_column
                if current_column == 0:
                    break

        assignments = sorted(
            (row - 1, column - 1)
            for column, row in enumerate(matched_row[1:], start=1)
            if row != 0
        )
        return [row for row, _ in assignments], [column for _, column in assignments]

    def _solve_score_assignment(self, db_coords, user_points, row_weights=None):
        row_count = len(db_coords)
        user_count = len(user_points)
        if row_count == 0:
            return {}
        if row_weights is None:
            row_weights = [1] * row_count

        # Match the 13C assignment objective: maximize in-tolerance matches,
        # then minimize total scaled displacement. Out-of-tolerance pairs are
        # retained when needed, matching the established 13C fallback behavior.
        cost = np.full((row_count, user_count + row_count), 1e6, dtype=np.float64)
        for row, db_point in enumerate(db_coords):
            for column, user_point in enumerate(user_points):
                distance = self._point_distance(db_point, user_point)
                in_tolerance = (
                    abs(db_point[0] - user_point[0]) <= self.c_tol
                    and abs(db_point[1] - user_point[1]) <= self.h_tol
                )
                cost[row, column] = (0.0 if in_tolerance else 1e4) + distance
            cost[row, user_count + row] = 1e6

        rows, columns = self._linear_sum_assignment(cost)
        return {
            row: column if column < user_count else None
            for row, column in zip(rows, columns)
        }

    def _process_single_molecule(self, mol_id, db_points, user_points, h_type=None):
        if self.CH_merge == "N":
            matches = []
            assigned = self._solve_score_assignment([p[2] for p in db_points], user_points)
            used_db = set()
            for db_idx, user_idx in assigned.items():
                if user_idx is None:
                    continue
                db_entry = db_points[db_idx]
                dist = self._point_distance(db_entry[2], user_points[user_idx])
                matches.append(
                    (
                        mol_id,
                        db_entry[0],
                        json.dumps(db_entry[2]),
                        json.dumps(user_points[user_idx]),
                        round(dist, 2),
                        h_type if h_type else db_entry[1],
                    )
                )
                used_db.add(db_idx)

            for idx, db_entry in enumerate(db_points):
                if idx not in used_db:
                    matches.append(
                        (
                            mol_id,
                            db_entry[0],
                            json.dumps(db_entry[2]),
                            json.dumps((0, 0)),
                            float("inf"),
                            h_type if h_type else db_entry[1],
                        )
                    )
            return matches
        if self.CH_merge == "Y":
            coord_groups = {}
            for idx, entry in enumerate(db_points):
                coord_groups.setdefault(tuple(entry[2]), []).append((idx, entry))

            group_coords = list(coord_groups)
            assigned = self._solve_score_assignment(
                group_coords,
                user_points,
                [len(coord_groups[coord]) for coord in group_coords],
            )
            matches = []
            matched_coords = set()
            for group_idx, user_idx in assigned.items():
                coord = group_coords[group_idx]
                if user_idx is None:
                    continue
                dist = round(self._point_distance(coord, user_points[user_idx]), 2)
                for _, entry in coord_groups[coord]:
                    c_index, h_type_val, _ = entry
                    matches.append(
                        (
                            mol_id,
                            c_index,
                            json.dumps(coord),
                            json.dumps(user_points[user_idx]),
                            dist,
                            h_type if h_type else h_type_val,
                        )
                    )

                matched_coords.add(coord)

            for coord, group in coord_groups.items():
                if coord in matched_coords:
                    continue
                for idx, entry in group:
                    c_index, h_type_val, _ = entry
                    matches.append(
                        (
                            mol_id,
                            c_index,
                            json.dumps(coord),
                            json.dumps((0, 0)),
                            float("inf"),
                            h_type if h_type else h_type_val,
                        )
                    )

            return matches

    def _match_mode1(self):
        """模式1：每个分子独立匹配所有用户点"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            molecules = {}
            for row in conn.execute(
                "SELECT mol_id, c_index, h_count, coordinates FROM carbon_h_coordinates"
            ):
                points = json.loads(row[3])
                if row[0] not in molecules:
                    molecules[row[0]] = []
                for p in points:
                    molecules[row[0]].append((row[1], row[2], p))

        all_matches = []
        user_pts = [tuple(p) for p in self.user_points["All_type"]]
        for mol_id in sorted(molecules.keys()):
            used_user = set()  # 每个分子独立使用
            db_points = molecules[mol_id]
            matches = self._process_single_molecule(
                mol_id=mol_id, db_points=db_points, user_points=user_pts
            )
            all_matches += matches
        return all_matches

    def _match_mode2(self):
        """新增：模式2专用匹配方法（合并处理type1/3）"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            # 获取所有type1和type3的原子
            molecules = {}
            for row in conn.execute("""
                SELECT mol_id, c_index, h_count, coordinates
                FROM carbon_h_coordinates
                WHERE h_count IN (1, 3)"""):
                points = json.loads(row[3])
                mol_id = row[0]
                if mol_id not in molecules:
                    molecules[mol_id] = []
                for p in points:
                    molecules[mol_id].append((row[1], row[2], p))  # (c_index, h_type, coordinates)

        all_matches = []
        user_pts = [tuple(p) for p in self.user_points["type13"]]

        for mol_id in sorted(molecules.keys()):
            used_user = set()  # 每个分子独立使用
            db_points = molecules[mol_id]

            # 处理type1/3组合
            matches = self._process_single_molecule(
                mol_id=mol_id, db_points=db_points, user_points=user_pts
            )
            all_matches += matches

        # 单独处理type2
        all_matches += self._match_by_type(h_type=2, user_points=self.user_points["type2"])
        return all_matches

    def _match_by_type(self, h_type, user_points):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            molecules = {}
            for row in conn.execute(
                """
                SELECT mol_id, c_index, coordinates
                FROM carbon_h_coordinates
                WHERE h_count = ?""",
                (h_type,),
            ):
                points = json.loads(row[2])
                if row[0] not in molecules:
                    molecules[row[0]] = []
                for p in points:
                    molecules[row[0]].append((row[1], p))

        all_matches = []
        user_pts = [tuple(p) for p in user_points]
        for mol_id in sorted(molecules.keys()):
            # 每个分子独立处理
            db_points = [(p[0], h_type, p[1]) for p in molecules[mol_id]]
            matches = self._process_single_molecule(
                mol_id=mol_id, db_points=db_points, user_points=user_pts, h_type=h_type
            )
            all_matches += matches
        return all_matches

    def process(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DELETE FROM user_matches")
            matches = []

            if self.mode == 1:
                matches = self._match_mode1()

            elif self.mode == 2:
                matches = self._match_mode2()
            elif self.mode == 3:
                # 分别处理三种类型
                matches += self._match_by_type(h_type=1, user_points=self.user_points["type1"])
                matches += self._match_by_type(h_type=2, user_points=self.user_points["type2"])
                matches += self._match_by_type(h_type=3, user_points=self.user_points["type3"])

            # 批量插入数据
            batch_size = 500
            with tqdm(total=len(matches), desc="Save the matching results") as pbar:
                for i in range(0, len(matches), batch_size):
                    batch = matches[i : i + batch_size]
                    conn.executemany(
                        """
                        INSERT INTO user_matches
                        (mol_id, c_index, db_point, user_point, distance, h_type)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        batch,
                    )
                    pbar.update(len(batch))

            # 统计结果
            matched = conn.execute(
                "SELECT COUNT(*) FROM user_matches WHERE distance < 1e9"
            ).fetchone()[0]
            print(
                f"Match complete! Valid matches {matched} and placeholder {len(matches) - matched} records"
            )


class MoleculeScoreCalculator:
    """分子评分计算器，处理CH坐标匹配结果的评分"""

    def __init__(self, db_path: str, c_tolerance: float, h_tolerance: float):
        """
        :param db_path: 数据库路径
        :param c_tolerance: 碳位移允许误差（ppm）
        :param h_tolerance: 氢位移允许误差（ppm，未放大前）
        """
        self.db_path = db_path
        self.c_tol = c_tolerance
        self.h_tol = h_tolerance
        self._create_score_table()

    def _create_score_table(self):
        """创建评分存储表"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DROP TABLE IF EXISTS molecule_ch_scores")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS molecule_ch_scores (
                    mol_id INTEGER PRIMARY KEY,
                    score REAL,
                    matched_points INTEGER,
                    total_points INTEGER,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)
            conn.commit()

    def _calculate_point_score(self, db_point: tuple, user_point: tuple) -> float:
        return _calculate_hsqc_point_score(
            db_point, user_point, self.c_tol, self.h_tol
        )

    def process_all(self):
        """批量处理所有分子"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            # 获取所有匹配数据
            cursor = conn.execute("""
                SELECT m.mol_id,
                       json_extract(u.db_point, '$[0]') as db_c,
                       json_extract(u.db_point, '$[1]') as db_h,
                       json_extract(u.user_point, '$[0]') as user_c,
                       json_extract(u.user_point, '$[1]') as user_h
                FROM user_matches u
                JOIN molecules m ON u.mol_id = m.mol_id
            """)

            # 按分子分组计算
            from collections import defaultdict

            mol_data = defaultdict(list)
            for row in cursor:
                db_point = (row["db_c"], row["db_h"])
                user_point = (row["user_c"], row["user_h"])
                mol_data[row["mol_id"]].append((db_point, user_point))

            # 计算评分
            scores = []
            for mol_id, points in mol_data.items():
                total_score = 0.0
                valid_points = 0

                for db_p, user_p in points:
                    score = self._calculate_point_score(db_p, user_p)
                    total_score += score
                    valid_points += 1 if user_p != (0, 0) else 0

                avg_score = total_score / len(points) if points else 0.0
                scores.append((mol_id, avg_score, valid_points, len(points)))

            # 批量存储
            conn.executemany(
                """
                INSERT OR REPLACE INTO molecule_ch_scores
                (mol_id, score, matched_points, total_points)
                VALUES (?, ?, ?, ?)
            """,
                scores,
            )
            conn.commit()
