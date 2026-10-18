"""Render the report and demo script from templates → reports/report.md, reports/demo_script.md.

uv run python scripts/build_report.py      (or: make report)

Every number comes from reports/tables (written by the pipeline scripts): the templates
(reports/report_template.md, reports/demo_template.md) hold the prose and `{{ expression }}`
fields that are evaluated against the tables below. Edit the templates, never the outputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from frminer.config import path

FIELD = re.compile(r"\{\{(.+?)\}\}", flags=re.S)
TEMPLATES = {"report_template.md": "report.md", "demo_template.md": "demo_script.md"}


# --- reading the tables -----------------------------------------------------------------


def md_tables(file: Path) -> list[pd.DataFrame]:
    """Every pipe table of a markdown file, numeric columns converted."""
    tables, block = [], []
    for line in [*file.read_text().splitlines(), ""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif block:
            rows = [r for r in block[1:] if not set("".join(r)) <= set(":-")]
            df = pd.DataFrame(rows, columns=block[0])
            for col in df.columns:
                num = pd.to_numeric(df[col].str.replace(",", "").str.rstrip("%"), errors="coerce")
                if num.notna().sum() == df[col].ne("").sum() and df[col].ne("").any():
                    df[col] = num / 100 if df[col].str.endswith("%").all() else num
            tables.append(df)
            block = []
    return tables


def grab(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"pattern not found in table: {pattern}")
    return cast(m.group(1))


def context(tables: Path) -> dict:
    read = lambda name: (tables / name).read_text()  # noqa: E731
    ctx: dict = {}

    ctx["corpus"] = json.loads(read("corpus_stats.json"))
    ctx["csum"] = md_tables(tables / "corpus_summary.md")[0].set_index("source")
    ctx["corpus_note"] = grab(r"Note: (.+)", read("corpus_summary.md"), str)

    gold_md = read("gold_summary.md")
    g = md_tables(tables / "gold_summary.md")
    ctx["gold"] = {
        "items": grab(r"(\d+) items", gold_md, int),
        "by_split": g[0].set_index("label"),
        "by_source": g[1].set_index("label"),
        "kappa": grab(r"Cohen's κ = ([\d.]+)", gold_md),
        "agreement": grab(r"raw agreement ([\d.]+)%", gold_md) / 100,
        "pilot_items": grab(r"(\d+) items · raw agreement", gold_md, int),
    }

    det = pd.read_csv(tables / "detect_test.csv").set_index("system")
    ctx["det"] = det
    ctx["dep"], ctx["teacher"] = det.loc["SetFit (silver)"], det.loc["LLM zero-shot"]
    ctx["dev"] = pd.read_csv(tables / "detect_dev.csv").set_index("method")
    ctx["per_class"] = md_tables(tables / "detect_test_per_class.md")[0].set_index("")
    err_md = read("detect_errors.md")
    e = md_tables(tables / "detect_errors.md")
    ctx["errors"] = {
        "deployed": grab(r"## SetFit \(silver\): (\d+) errors", err_md, int),
        "teacher": grab(r"## LLM zero-shot: (\d+) errors", err_md, int),
        "both_wrong": grab(r"Both systems are wrong on (\d+) items", err_md, int),
        "same_label": grab(r"on (\d+) of them they agree", err_md, int),
        "top": e[0],
    }

    ext = md_tables(tables / "extract_eval.md")[0]
    ctx["ext"] = ext.set_index(["setting", "method"])

    grid_md = read("cluster_grid.md")
    grid = md_tables(tables / "cluster_grid.md")[0]
    mcs = grab(r"min_cluster_size=(\d+)", grid_md, int)
    ms = grab(r"min_samples=(\d+)\.", grid_md, int)
    chosen = grid[(grid["min_cluster_size"] == mcs) & (grid["min_samples"] == ms)].iloc[0]
    ctx["grid"] = {
        "phrases": grab(r"(\d+) distinct phrases", grid_md, int),
        "mentions": grab(r"\((\d+) mentions\)", grid_md, int),
        "settings": len(grid),
        "min_cluster_size": mcs,
        "min_samples": ms,
        "clusters_before": int(chosen["n_clusters"]),
        "proposed_groups": grab(r"proposed (\d+) groups", grid_md, int),
        "proposed_merges": grab(r"\((\d+) merges\)", grid_md, int),
        "confirmed": grab(r"confirmed (\d+) of them", grid_md, int),
        "joined": grab(r"joined (\d+) clusters", grid_md, int),
        "into": grab(r"clusters into (\d+)", grid_md, int),
        "themes": grab(r"leaving (\d+) themes", grid_md, int),
    }
    ctx["cmp"] = md_tables(tables / "cluster_compare.md")[0].set_index(
        md_tables(tables / "cluster_compare.md")[0]["method"].str[0]
    )

    pur_md = read("cluster_eval.md")
    pur = md_tables(tables / "cluster_eval.md")[0]
    pur["key"] = pur["method"].str[0] + " " + pur["rater"]
    ctx["pur"] = pur.set_index("key")
    ctx["jh"] = {
        "members": grab(r"(\d+) members: agreement", pur_md, int),
        "agreement": grab(r"agreement ([\d.]+), Cohen", pur_md),
        "kappa": grab(r"Cohen's κ ([\d.]+)", pur_md),
        "judge_intruders": grab(r"judge calls (\d+)% intruders", pur_md) / 100,
        "human_intruders": grab(r"the human (\d+)%", pur_md) / 100,
        "name_yes": grab(r"yes (\d+)%", pur_md) / 100,
        "name_partly": grab(r"partly (\d+)%", pur_md) / 100,
    }

    ctx["top"] = md_tables(tables / "top_themes.md")[0]
    trends_md = read("trends.md")
    ctx["trends"] = {
        "recent_items": grab(r"Last \d+ weeks \((\d+) corpus items\)", trends_md, int),
        "prev_items": grab(r"weeks before \((\d+)\)", trends_md, int),
        "weeks": grab(r"Last (\d+) weeks", trends_md, int),
        "emerging": grab(r"## Emerging \((\d+)\)", trends_md, int),
        "declining": grab(r"## Declining \((\d+)\)", trends_md, int),
        "declining_table": md_tables(tables / "trends.md")[0],
    }

    ins_md = read("insights.md")
    ins = md_tables(tables / "insights.md")
    ctx["ins"] = {
        "reddit_leaning": ins[0],
        "play_leaning": ins[1],
        "angriest": ins[2],
        "happiest": ins[3],
        "competitors": ins[4],
        "status": {
            s: grab(rf"{s}: (\d+) themes", ins_md, int)
            for s in ["new", "premium_only", "partly", "exists"]
        },
        "exists": ins[6],
        "premium_only": ins[7],
        "partly": ins[8],
    }
    return ctx


# --- formatting helpers available inside templates --------------------------------------


def n(x) -> str:
    return f"{int(x):,}"


def f2(x) -> str:
    return f"{float(x):.2f}"


def f3(x) -> str:
    return f"{float(x):.3f}"


def pct(x, digits: int = 0) -> str:
    return f"{float(x):.{digits}%}"


def md(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    rename: dict | None = None,
    pct_cols: tuple[str, ...] = (),
    digits: int = 3,
) -> str:
    """A table for the report: floats rounded, thousands separators, chosen columns as %."""
    out = (df[cols] if cols else df).copy()
    for col in out.columns:
        if col in pct_cols:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: (
                    ""
                    if pd.isna(v)
                    else f"{v:.{digits}g}"
                    if abs(v) < 1e-3 and v != 0
                    else f"{v:.{digits}f}".rstrip("0").rstrip(".")
                )
            )
    out = out.replace({"nan": "", "None": ""})
    return out.rename(columns=rename or {}).to_markdown(index=False, intfmt=",")


def names(df: pd.DataFrame, k: int = 5, col: str = "name") -> str:
    vals = list(df[col].head(k))
    return ", ".join(vals[:-1]) + f" and {vals[-1]}" if len(vals) > 1 else "".join(vals)


HELPERS = {
    "n": n,
    "f2": f2,
    "f3": f3,
    "pct": pct,
    "md": md,
    "names": names,
    "len": len,
    "int": int,
    "float": float,
    "round": round,
    "max": max,
    "min": min,
}


def render(template: str, ctx: dict) -> str:
    env = {**HELPERS, **ctx}

    def field(m: re.Match) -> str:
        expr = m.group(1).strip()
        try:
            return str(eval(expr, {"__builtins__": {}, **env}))  # trusted, repo-owned templates
        except Exception as e:
            raise ValueError(f"template field {{{{ {expr} }}}} failed: {e}") from e

    return FIELD.sub(field, template)


def main() -> None:
    reports = path("reports")
    ctx = context(reports / "tables")
    for src, dst in TEMPLATES.items():
        body = render((reports / src).read_text(), ctx)
        header = (
            f"<!-- Generated by scripts/build_report.py from reports/{src}; "
            "edit the template, not this file. -->\n\n"
        )
        (reports / dst).write_text(header + body)
        print(f"wrote reports/{dst}")


if __name__ == "__main__":
    main()
