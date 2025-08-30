# loaders_employee_csv.py（新規 or 既存のローダーファイルに追記）

import os
import csv
from collections import defaultdict
from typing import List
from langchain_core.documents import Document

EMPLOYEE_CSV_FILENAME = "社員名簿.csv"

_DEPT_KEYS = ["部署", "部門", "部", "Department", "dept"]  # 部署列の候補

def _read_csv_rows(path: str):
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with open(path, encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                return rows
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSVの読み込みに失敗しました: {path}. 最後のエラー: {last_err}")

def _detect_dept_col(rows: List[dict]) -> str:
    if not rows:
        return "部署"
    sample = rows[0].keys()
    for k in _DEPT_KEYS:
        if k in sample:
            return k
    # 見つからない場合のフォールバック（不明部署扱い）
    return None

def _format_department_doc(dept: str, members: List[dict], dept_col: str) -> Document:
    # 表示順の優先候補（存在する列のみ使う）
    preferred_order = ["氏名", "社員ID", "役職", "役位", "メール", "内線", "拠点", "勤務地", "入社日"]
    sample_cols = list(members[0].keys())
    cols = [c for c in preferred_order if c in sample_cols]
    # 部署列は重複なので除外
    extra = [c for c in sample_cols if c not in cols and c != dept_col]
    cols += extra

    # 部署見出し + 同義語（検索当たりを良くするため）
    header = f"《部署》{dept}"
    if dept in ("人事部", "人事", "HR", "Human Resources"):
        header += "（人事部 / HR / Human Resources）"

    lines = []
    for i, r in enumerate(members, 1):
        parts = [f"{c}:{(r.get(c) or '').strip()}" for c in cols if (r.get(c) or "").strip()]
        lines.append(f"{i}. " + " | ".join(parts))

    body = "\n".join(lines)
    keywords = f"\nキーワード: {dept}, {dept}の社員一覧, {dept} メンバー, 従業員, 社員, 社内名簿"
    content = f"{header}\n人数: {len(members)}\n{body}{keywords}"

    return Document(
        page_content=content,
        metadata={
            "source": "社員名簿.csv",
            "doctype": "employee_csv",
            "department": dept,
            "merged": True,
        },
    )

def load_employee_csv_grouped_by_department(path: str) -> List[Document]:
    rows = _read_csv_rows(path)
    dept_col = _detect_dept_col(rows)
    groups = defaultdict(list)
    for r in rows:
        dept = (r.get(dept_col) if dept_col else None) or "不明"
        groups[dept].append(r)

    docs = []
    for dept, members in groups.items():
        docs.append(_format_department_doc(dept, members, dept_col or "部署"))
    return docs

class EmployeeCSVDepartmentLoader:
    """社員名簿.csv専用：部署ごとに1ドキュメントへ統合するローダー（.load() を実装）"""
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Document]:
        return load_employee_csv_grouped_by_department(self.path)
