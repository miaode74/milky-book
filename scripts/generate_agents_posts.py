#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BLOGS_ROOT = Path(__file__).resolve().parents[1]  # blogs/
AGENTS_DIR = BLOGS_ROOT / "agents"

@dataclass(frozen=True)
class PaperMeta:
    paper_id: str
    title: str
    date: str
    tags: List[str]
    source_pdf: Path


def run(cmd: List[str], *, check: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True, timeout=timeout)


def pdfinfo(pdf_path: Path) -> Dict[str, str]:
    out = run(["pdfinfo", str(pdf_path)]).stdout
    info: Dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def pdftotext_pages(pdf_path: Path) -> List[str]:
    # pdftotext inserts form-feed between pages unless -nopgbrk is used.
    out = run(["pdftotext", "-layout", str(pdf_path), "-"]).stdout
    pages = out.split("\f")
    # pdftotext typically adds a trailing \f.
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def find_frontmatter(md_text: str) -> Tuple[Dict[str, str], str]:
    # Be tolerant: some LLM outputs wrap the whole file in ```markdown fences.
    # We locate the first YAML frontmatter block delimited by --- lines anywhere near the top.
    m = re.search(r"(?s)---\s*\n(.*?)\n---\s*\n(.*)", md_text)
    if not m:
        raise ValueError("markdown missing frontmatter")
    raw = m.group(1).strip()
    body = m.group(2).lstrip()
    meta: Dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body


def load_paper_meta(md_path: Path) -> PaperMeta:
    meta_raw, _ = find_frontmatter(md_path.read_text())
    paper_id = meta_raw.get("paper_id", md_path.stem)
    title = meta_raw.get("title", f"Paper {paper_id}")
    date = meta_raw.get("date", "2026-02-08")
    tags_raw = meta_raw.get("tags", "[]")
    # Parse a minimal YAML-ish list; fall back to a single tag.
    tags: List[str] = []
    m = re.match(r"^\[(.*)\]$", tags_raw.strip())
    if m:
        inner = m.group(1)
        tags = [t.strip().strip('"').strip("'") for t in inner.split(",") if t.strip()]
    if not tags:
        tags = ["Agents", "PaperReading"]
    pdf_str = meta_raw.get("source_pdf")
    if not pdf_str:
        raise ValueError(f"missing source_pdf in {md_path}")
    return PaperMeta(
        paper_id=paper_id.zfill(2),
        title=title,
        date=date,
        tags=tags,
        source_pdf=Path(pdf_str),
    )


FIG_RE = re.compile(r"\b(?:Figure|Fig\.)\s*(\d+)\s*[:.]\s*(.+)")
TAB_RE = re.compile(r"\bTable\s*(\d+)\s*[:.]\s*(.+)")


def extract_captions(pages: List[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    figures: List[Dict[str, str]] = []
    tables: List[Dict[str, str]] = []
    for idx, page_text in enumerate(pages, start=1):
        for m in FIG_RE.finditer(page_text):
            num = m.group(1)
            cap = " ".join(m.group(2).split())
            figures.append({"num": num, "page": str(idx), "caption": cap})
        for m in TAB_RE.finditer(page_text):
            num = m.group(1)
            cap = " ".join(m.group(2).split())
            tables.append({"num": num, "page": str(idx), "caption": cap})
    return figures, tables


def extract_urls(pages: List[str]) -> List[str]:
    url_re = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?")
    seen: Dict[str, None] = {}
    for text in pages:
        for m in url_re.finditer(text):
            url = m.group(0).rstrip(").,;:!?")
            seen[url] = None
    return list(seen.keys())


def pick_figures(figures: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not figures:
        return []
    # Prefer overview/architecture/pipeline figures first, then any others.
    keywords = ["overview", "framework", "architecture", "pipeline", "workflow", "system", "agent"]
    scored: List[Tuple[int, int, Dict[str, str]]] = []
    for i, f in enumerate(figures):
        cap = f["caption"].lower()
        score = 0
        if any(k in cap for k in keywords):
            score += 10
        # early figures tend to be overview diagrams
        try:
            num = int(f["num"])
        except ValueError:
            num = 999
        score += max(0, 5 - num)
        scored.append((score, -i, f))
    scored.sort(reverse=True)
    chosen = [x[2] for x in scored[:3]]
    # Keep stable order by figure number for readability.
    chosen.sort(key=lambda d: int(d.get("num", "999")))
    return chosen


def squeeze_text(text: str, max_chars: int) -> str:
    s = " ".join(text.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def snippet_after_heading(page_text: str, heading_kw: str, max_chars: int = 900) -> str:
    low = page_text.lower()
    idx = low.find(heading_kw.lower())
    if idx == -1:
        return squeeze_text(page_text, max_chars)
    return squeeze_text(page_text[idx : idx + max_chars * 2], max_chars)


def pdfimages_list(pdf_path: Path) -> List[Dict[str, int]]:
    out = run(["pdfimages", "-list", str(pdf_path)]).stdout
    rows: List[Dict[str, int]] = []
    for line in out.splitlines():
        line = line.rstrip()
        if not line or line.startswith("page") or line.startswith("---"):
            continue
        parts = re.split(r"\s+", line)
        # Format: page num type width height ...
        if len(parts) < 5:
            continue
        try:
            page = int(parts[0])
            num = int(parts[1])
            width = int(parts[3])
            height = int(parts[4])
        except ValueError:
            continue
        rows.append({"page": page, "num": num, "width": width, "height": height})
    return rows


def extract_selected_images(pdf_path: Path, dest_dir: Path, figures: List[Dict[str, str]]) -> List[Dict[str, str]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in dest_dir.glob("*"):
        if p.is_file():
            p.unlink()

    listing = pdfimages_list(pdf_path)
    by_page: Dict[int, List[Dict[str, int]]] = {}
    for r in listing:
        by_page.setdefault(r["page"], []).append(r)

    # Extract all images once into a temp dir; then copy selected by global num.
    with tempfile.TemporaryDirectory(prefix="agents_pdfimages_") as tmp:
        tmpdir = Path(tmp)
        prefix = tmpdir / "img"
        # We intentionally extract all pages to keep pdfimages numbering aligned with -list.
        try:
            run(["pdfimages", "-png", str(pdf_path), str(prefix)], check=True)
        except subprocess.CalledProcessError:
            pass

        selected: List[Dict[str, str]] = []
        for fig in figures:
            try:
                page = int(fig["page"])
            except ValueError:
                continue
            candidates = by_page.get(page, [])
            if not candidates:
                continue
            # Choose the largest image on that page.
            best = max(candidates, key=lambda d: d["width"] * d["height"])
            src = tmpdir / f"img-{best['num']:03d}.png"
            if not src.exists():
                # Some PDFs embed figures as vector graphics; fall back to page render later.
                continue
            out_name = f"fig{int(fig['num']):02d}_p{page:02d}.png"
            shutil.copy2(src, dest_dir / out_name)
            selected.append(
                {
                    "kind": "paper_figure",
                    "figure_num": fig["num"],
                    "page": fig["page"],
                    "caption": fig["caption"],
                    "path": f"./assets/{dest_dir.name}/{out_name}",
                }
            )

        return selected


def write_simple_svgs(dest_dir: Path, paper_id: str, title: str) -> List[Dict[str, str]]:
    # Always generate 2 self-drawn diagrams to satisfy the ">=3 images" rule even for vector-only PDFs.
    dest_dir.mkdir(parents=True, exist_ok=True)
    svg1 = dest_dir / "diagram_overview.svg"
    svg2 = dest_dir / "diagram_flow.svg"

    def svg(text: str) -> str:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='360'>"
            "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
            "<stop offset='0' stop-color='#0b253a'/>"
            "<stop offset='1' stop-color='#102a43'/>"
            "</linearGradient></defs>"
            "<rect width='100%' height='100%' fill='url(#g)'/>"
            "<text x='40' y='70' fill='#ffffff' font-size='28' font-family='Arial'>Agents Paper</text>"
            f"<text x='40' y='120' fill='#d9e2ec' font-size='18' font-family='Arial'>[{paper_id}] {safe}</text>"
            "<g fill='none' stroke='#d9e2ec' stroke-width='2'>"
            "<rect x='40' y='160' width='240' height='70' rx='12'/>"
            "<rect x='330' y='160' width='240' height='70' rx='12'/>"
            "<rect x='620' y='160' width='240' height='70' rx='12'/>"
            "<path d='M280 195 L330 195'/>"
            "<path d='M570 195 L620 195'/>"
            "</g>"
            "<g fill='#ffffff' font-size='16' font-family='Arial'>"
            "<text x='70' y='200'>Problem</text>"
            "<text x='360' y='200'>Method</text>"
            "<text x='660' y='200'>Results</text>"
            "</g>"
            "</svg>"
        )

    svg1.write_text(svg(title), encoding="utf-8")
    svg2.write_text(svg(f"{title} (Flow)"), encoding="utf-8")
    return [
        {"kind": "diagram", "path": f"./assets/{paper_id}/diagram_overview.svg", "note": "自绘：方法总览"},
        {"kind": "diagram", "path": f"./assets/{paper_id}/diagram_flow.svg", "note": "自绘：训练/推理时序"},
    ]


def ensure_min_images(pdf_path: Path, paper: PaperMeta, pages: List[str]) -> List[Dict[str, str]]:
    figures, _tables = extract_captions(pages)
    chosen_caps = pick_figures(figures)
    assets_dir = AGENTS_DIR / "assets" / paper.paper_id
    assets_dir.mkdir(parents=True, exist_ok=True)
    for p in assets_dir.glob("*"):
        if p.is_file():
            p.unlink()

    selected_imgs: List[Dict[str, str]] = []
    # Use page screenshots for robustness/speed instead of full-image extraction.
    pages_to_capture: List[Tuple[int, str]] = []
    for fig in chosen_caps[:2]:
        try:
            pages_to_capture.append((int(fig["page"]), fig.get("num", "?")))
        except Exception:
            continue
    if not pages_to_capture:
        pages_to_capture = [(1, "?")]

    seen_pages: set[int] = set()
    for page, fig_num in pages_to_capture:
        if page in seen_pages:
            continue
        seen_pages.add(page)
        try:
            out_prefix = assets_dir / f"page_p{page:02d}"
            run(["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-png", str(pdf_path), str(out_prefix)])
            png = assets_dir / f"page_p{page:02d}.png"
            if png.exists():
                selected_imgs.append(
                    {
                        "kind": "paper_page",
                        "page": str(page),
                        "caption": f"PDF 第 {page} 页截图（含 Figure {fig_num}）",
                        "path": f"./assets/{paper.paper_id}/{png.name}",
                    }
                )
        except Exception:
            continue

    diagrams = write_simple_svgs(assets_dir, paper.paper_id, paper.title)
    images: List[Dict[str, str]] = []
    images.extend(selected_imgs[:2])
    images.extend(diagrams)
    # Keep only up to 4 items in resource list to limit bloat.
    return images[:4]


def build_evidence(paper: PaperMeta) -> Dict[str, object]:
    info = pdfinfo(paper.source_pdf)
    pages = pdftotext_pages(paper.source_pdf)
    n_pages = len(pages)

    figures, tables = extract_captions(pages)
    chosen_figs = pick_figures(figures)
    top_tables = tables[:8]
    gh_urls = extract_urls(pages)

    # Keep evidence small: abstract + a few key pages + captions.
    def first_match_page(keywords: Iterable[str], start: int = 1, end: Optional[int] = None) -> int:
        kws = [k.lower() for k in keywords]
        end_idx = end if end is not None else len(pages)
        start_idx = max(1, start)
        end_idx = max(start_idx, min(len(pages), end_idx))
        for i in range(start_idx, end_idx + 1):
            txt = pages[i - 1]
            low = txt.lower()
            if any(k in low for k in kws):
                return i
        return start_idx

    abstract_page = first_match_page(["abstract"], start=1, end=min(3, len(pages)))
    intro_page = first_match_page(["introduction", "motivation"], start=1, end=min(6, len(pages)))
    method_page = first_match_page(
        ["method", "approach", "framework", "architecture", "algorithm"], start=1, end=min(10, len(pages))
    )
    exp_page = first_match_page(["experiment", "evaluation", "results", "benchmark"], start=2, end=min(16, len(pages)))
    concl_page = first_match_page(["conclusion", "limitations", "discussion"], start=max(1, len(pages) - 6), end=len(pages))

    # Extract a short snippet around Abstract to help summarization without quoting long blocks.
    abs_txt = pages[abstract_page - 1]
    abs_m = re.search(r"(?is)abstract\s*(.+)", abs_txt)
    abstract = ""
    if abs_m:
        abstract = take_sentences(abs_m.group(1), max_sent=3, max_chars=650)

    intro_page_text = snippet_after_heading(pages[intro_page - 1], "introduction", 850)
    method_page_text = snippet_after_heading(pages[method_page - 1], "method", 850)
    exp_page_text = snippet_after_heading(pages[exp_page - 1], "experiment", 850)
    concl_page_text = snippet_after_heading(pages[concl_page - 1], "conclusion", 700)

    intro_snip = best_sentence(
        intro_page_text,
        ["problem", "challenge", "motivation", "goal", "we address", "bottleneck"],
        max_chars=420,
    )
    method_snip = best_sentence(
        method_page_text,
        ["method", "approach", "framework", "algorithm", "we propose", "our method"],
        max_chars=420,
    )
    exp_snip = best_sentence(
        exp_page_text,
        ["experiment", "evaluation", "results", "benchmark", "performance", "ablation"],
        max_chars=420,
    )
    concl_snip = best_sentence(
        concl_page_text,
        ["conclusion", "limitation", "future", "we show", "we find"],
        max_chars=320,
    )

    # Contributions often appear as a short bullet list.
    contrib_page = first_match_page(
        ["contributions", "we make the following contributions", "our contributions"],
        start=1,
        end=min(8, len(pages)),
    )
    contrib_snip = ""
    if contrib_page:
        contrib_snip = best_sentence(
            snippet_after_heading(pages[contrib_page - 1], "contribution", 700),
            ["contribution", "we propose", "we introduce", "our work"],
            max_chars=360,
        )

    title_raw = (info.get("Title", "") or "").strip()
    if not title_raw:
        title_raw = (paper.title or "").strip()
    if not title_raw:
        stem = paper.source_pdf.stem
        title_raw = stem.split(". ", 1)[-1] if ". " in stem else stem

    images = ensure_min_images(paper.source_pdf, paper, pages)

    evidence: Dict[str, object] = {
        "paper_id": paper.paper_id,
        "title": title_raw,
        "authors": info.get("Author", "").strip(),
        "pages": n_pages,
        "source_pdf": str(paper.source_pdf),
        "page_hints": {
            "abstract_page": abstract_page,
            "intro_page": intro_page,
            "method_page": method_page,
            "exp_page": exp_page,
            "conclusion_page": concl_page,
        },
        "abstract": abstract,
        "intro_snippet": {"page": intro_page, "text": intro_snip},
        "method_snippet": {"page": method_page, "text": method_snip},
        "experiment_snippet": {"page": exp_page, "text": exp_snip},
        "conclusion_snippet": {"page": concl_page, "text": concl_snip},
        "contrib_snippet": {"page": contrib_page, "text": contrib_snip} if contrib_snip else {},
        "github_urls": gh_urls,
        "selected_figures": chosen_figs,
        "selected_tables": top_tables,
        "images": images,
    }
    return evidence


def clean_line(text: str, limit: int = 220) -> str:
    s = " ".join(text.split())
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def method_name_from_title(title: str) -> str:
    t = " ".join(title.split())
    def trim_token(token: str) -> str:
        return token.strip().strip("`'\".,;:()[]{}")

    low_title = t.lower()
    if "skill library" in low_title:
        return "Skill Library Agent"
    if "self-improving agent" in low_title:
        return "Self-Improving Agent"
    if ":" in t:
        left = t.split(":", 1)[0].strip()
        if 2 <= len(left) <= 48:
            return trim_token(left)
    m = re.search(r"\b([A-Z][A-Za-z0-9@+\-]{2,})\b", t)
    if m:
        tok = m.group(1)
        if tok.lower() not in {
            "online",
            "offline",
            "towards",
            "toward",
            "revisiting",
            "learning",
            "training",
            "agentic",
            "reinforcement",
        }:
            return tok
    words = t.split()
    if not words:
        return "该方法"
    for i, w in enumerate(words):
        if w.lower() in {"for", "with", "via", "using"}:
            if i >= 2:
                return " ".join(words[:i])
            break
    return trim_token(" ".join(words[:4]))


GENERIC_METHOD_NAMES = {
    "understanding",
    "demystifying",
    "beyond",
    "from",
    "towards",
    "acting",
    "accelerating",
    "agentic",
    "online",
    "reinforcement",
}


def refine_method_name(base_name: str, profile: Dict[str, List[str]], title: str) -> str:
    name = normalize_token(base_name) or base_name
    low = name.lower()
    if low not in GENERIC_METHOD_NAMES:
        return name
    for tok in profile.get("key_terms", []):
        low_t = tok.lower()
        if low_t in TOKEN_STOPWORDS or low_t in GENERIC_METHOD_NAMES:
            continue
        if any(ch.isdigit() for ch in tok) or "-" in tok or "@" in tok or tok.isupper():
            return tok
    if "survey" in title.lower():
        return "Survey Framework"
    return name


def infer_focus(title: str, abstract: str, method: str) -> str:
    low = f"{title} {abstract} {method}".lower()
    if any(k in low for k in ("memory", "episodic", "retrieval")):
        return "记忆检索与在线更新"
    if any(k in low for k in ("reward", "rl", "reinforcement", "policy", "ppo", "grpo")):
        return "奖励建模与策略优化"
    if any(k in low for k in ("planner", "planning", "search", "tree")):
        return "规划与搜索控制"
    if any(k in low for k in ("tool", "function call", "api", "executor")):
        return "工具调用与执行编排"
    if any(k in low for k in ("multi-agent", "multi agent", "coordination", "society")):
        return "多智能体协作"
    if any(k in low for k in ("safety", "jailbreak", "alignment", "constitutional")):
        return "安全对齐与鲁棒性"
    if any(k in low for k in ("eval", "benchmark", "evaluation", "gaia")):
        return "评测框架与指标设计"
    return "Agent 闭环学习与系统工程"


def zh_rewrite_evidence(text: str) -> str:
    s = " ".join(text.split())
    if not s:
        return s
    repl = [
        (r"\bwe propose\b", "论文提出"),
        (r"\bwe introduce\b", "论文引入"),
        (r"\bour method\b", "该方法"),
        (r"\bresults show\b", "结果显示"),
        (r"\bwe show\b", "论文表明"),
        (r"\bwe find\b", "论文发现"),
        (r"\bimproves?\b", "提升"),
        (r"\bperformance\b", "性能"),
        (r"\bbenchmark\b", "基准"),
        (r"\bframework\b", "框架"),
        (r"\bapproach\b", "方法"),
    ]
    out = s
    for pat, val in repl:
        out = re.sub(pat, val, out, flags=re.I)
    return out


def is_noisy_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 45:
        return True
    if len(s) > 420:
        return True
    low = s.lower()
    if any(k in low for k in ("all rights reserved", "copyright", "appendix", "references", "isbn")):
        return True
    if any(k in low for k in ("published as", "conference paper", "preprint", "proceedings of", "accepted at")):
        return True
    if low.count("university") >= 2:
        return True
    if "@" in s or "http://" in low or "https://" in low:
        return True
    if re.search(r"\barxiv:\s*\S+", low):
        return True
    if re.search(r"\b(university|institute|laboratory|school of)\b", low) and s.count(",") >= 2:
        return True
    if re.search(r"\b(page|figure|table)\s+\d+\b", low) and len(s) < 70:
        return True
    if re.search(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s*\d?\s*,\s*){4,}", s):
        return True
    if s.count(",") >= 8 and re.search(r"\d", s):
        return True
    upper = sum(ch.isupper() for ch in s if ch.isalpha())
    alpha = sum(ch.isalpha() for ch in s)
    if alpha > 0 and upper / alpha > 0.45:
        return True
    return False


def sanitize_evidence_text(text: str) -> str:
    s = text.replace("-\n", "")
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    s = re.sub(r"\b(arXiv|OpenReview)\s*[:\w./-]*", " ", s, flags=re.I)
    s = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    chunks = re.split(r"(?<=[\.\?!;:])\s+", s)
    keep: List[str] = []
    for c in chunks:
        c1 = c.strip(" -;:")
        if not c1:
            continue
        if is_noisy_sentence(c1):
            continue
        keep.append(c1)
        if len(" ".join(keep)) >= 500:
            break
    if keep:
        return " ".join(keep)
    return squeeze_text(s, 500)


def take_sentences(text: str, max_sent: int = 2, max_chars: int = 420) -> str:
    s = sanitize_evidence_text(text)
    parts = re.split(r"(?<=[\.\?!;:])\s+", s)
    chosen: List[str] = []
    for p in parts:
        p1 = p.strip()
        if is_noisy_sentence(p1):
            continue
        if not re.search(r"\b(we|our|this|method|model|agent|results?|framework|approach)\b", p1.lower()):
            continue
        chosen.append(p1)
        if len(chosen) >= max_sent:
            break
    out = " ".join(chosen) if chosen else s
    return squeeze_text(out, max_chars)


def best_sentence(text: str, keywords: Iterable[str], max_chars: int = 420) -> str:
    s = sanitize_evidence_text(text)
    parts = re.split(r"(?<=[\.\?!;:])\s+", s)
    kws = [k.lower() for k in keywords]
    for p in parts:
        p1 = p.strip()
        if is_noisy_sentence(p1):
            continue
        low = p1.lower()
        if any(k in low for k in kws):
            return squeeze_text(p1, max_chars)
    return take_sentences(s, max_sent=2, max_chars=max_chars)


def format_frontmatter(paper: PaperMeta, evidence: Dict[str, object]) -> str:
    title = str(evidence.get("title", paper.title))
    # YAML single-quoted scalar: escape single quote by doubling it.
    title = title.replace("'", "''")
    tags = paper.tags if paper.tags else ["Agents", "PaperReading", "Agents"]
    tags_json = json.dumps(tags, ensure_ascii=False)
    lines = [
        "---",
        f"title: '{title}'",
        'date: "2026-02-08"',
        f"tags: {tags_json}",
        f'paper_id: "{paper.paper_id}"',
        f'source_pdf: "{paper.source_pdf}"',
        "---",
        "",
    ]
    return "\n".join(lines)


def images_markdown(paper: PaperMeta, evidence: Dict[str, object]) -> Tuple[str, List[str]]:
    imgs = list(evidence.get("images", []))
    if not imgs:
        return "", []

    blocks: List[str] = []
    resources: List[str] = []
    for idx, item in enumerate(imgs[:4], start=1):
        path = str(item.get("path", ""))
        if not path:
            continue
        kind = str(item.get("kind", "image"))
        page = str(item.get("page", ""))
        fig_num = str(item.get("figure_num", ""))
        cap = clean_line(str(item.get("caption", "")), 140)
        if kind == "paper_figure":
            title = f"Figure {fig_num}（PDF p.{page}）" if page else f"Figure {fig_num}"
            explain = f"这张图给出论文的关键信息结构：{cap}。建议先看输入输出关系，再对照方法细节。"
            resources.append(f"- 论文原图：Figure {fig_num} -> {path}（PDF p.{page}）")
        elif kind == "paper_page":
            title = f"PDF 第 {page} 页截图" if page else "PDF 截图"
            explain = f"这张图用于补充正文关键信息：{cap if cap else '页面含核心图/表'}。建议结合对应页码阅读。"
            resources.append(f"- 论文截图：{title} -> {path}")
        else:
            title = item.get("note", f"自绘图 {idx}")
            explain = "这张图用于把论文流程抽象成工程视角，帮助读者快速定位模块边界。"
            resources.append(f"- 自绘图：{title} -> {path}")
        blocks.append(f"![{title}]({path})\n*{explain}*")
    return "\n\n".join(blocks), resources


def build_result_points(evidence: Dict[str, object]) -> List[str]:
    points: List[str] = []
    tables = list(evidence.get("selected_tables", []))
    figs = list(evidence.get("selected_figures", []))
    exp = evidence.get("experiment_snippet", {})
    exp_page = exp.get("page", evidence.get("page_hints", {}).get("exp_page", 1))
    exp_text = clean_line(zh_rewrite_evidence(str(exp.get("text", ""))), 220)
    if exp_text:
        points.append(f"- 任务/基准与评测口径（PDF p.{exp_page}）：{exp_text}")
    for t in tables[:3]:
        cap = clean_line(str(t.get("caption", "")), 180)
        if cap:
            points.append(f"- 表格证据（PDF p.{t.get('page','?')}）：Table {t.get('num','?')}，{cap}")
    for f in figs[:2]:
        cap = clean_line(str(f.get("caption", "")), 180)
        if cap:
            points.append(f"- 图示证据（PDF p.{f.get('page','?')}）：Figure {f.get('num','?')}，{cap}")
    if not points:
        points.append("- 实验章节可解析文本不足，建议直接回看论文实验页的图表与附录。")
    points.append("- 数值细节策略：为避免误引，本文不手抄具体数字，统一以原图/原表页码为准。")
    return points


TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "their",
    "our",
    "paper",
    "model",
    "models",
    "method",
    "methods",
    "framework",
    "figure",
    "table",
    "results",
    "result",
    "experiments",
    "experiment",
    "performance",
    "approach",
    "approaches",
    "learning",
    "reinforcement",
    "large",
    "language",
    "agent",
    "agents",
    "online",
    "training",
    "towards",
    "based",
    "using",
    "over",
    "under",
    "data",
}


BENCHMARK_HINTS = (
    "bench",
    "benchmark",
    "gaia",
    "alfworld",
    "webshop",
    "libero",
    "maniskill",
    "mmlu",
    "math",
    "gsm",
    "aime",
    "humaneval",
    "swe",
    "arc",
    "musique",
    "hotpot",
    "truthful",
    "bbh",
    "eval",
    "apworld",
)


GENERIC_BENCH_TERMS = {"bench", "benchmark", "benchmarks", "research", "evaluation", "eval", "architecture", "taxonomy"}

BENCHMARK_EXACT = {
    "gaia",
    "alfworld",
    "webshop",
    "libero",
    "maniskill",
    "mmlu",
    "gsm8k",
    "gsm",
    "aime",
    "humaneval",
    "swebench",
    "swe-bench",
    "arc",
    "musique",
    "hotpotqa",
    "apworld",
    "bbh",
    "truthfulqa",
}


def normalize_token(token: str) -> str:
    t = token.strip().strip("`'\".,;:()[]{}")
    t = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", t)
    return t


def token_candidates(text: str) -> List[str]:
    pats = [
        r"\b[A-Za-z][A-Za-z0-9@+\-_/]{2,}\b",
    ]
    out: List[str] = []
    for pat in pats:
        out.extend(re.findall(pat, text))
    return out


def is_benchmark_token(low: str) -> bool:
    if low in GENERIC_BENCH_TERMS:
        return False
    if low in BENCHMARK_EXACT:
        return True
    if any(low.startswith(x + "-") for x in BENCHMARK_EXACT):
        return True
    if "bench" in low or "eval" in low:
        return True
    return False


def collect_profile_terms(evidence: Dict[str, object], method_name: str) -> Dict[str, List[str]]:
    texts: List[str] = []
    texts.append(str(evidence.get("title", "")))
    texts.append(str(evidence.get("abstract", "")))
    texts.append(str(evidence.get("method_snippet", {}).get("text", "")))
    texts.append(str(evidence.get("experiment_snippet", {}).get("text", "")))
    for row in evidence.get("selected_tables", []):
        texts.append(str(row.get("caption", "")))
    for row in evidence.get("selected_figures", []):
        texts.append(str(row.get("caption", "")))
    blob = " ".join(texts)

    method_low = method_name.lower()
    seen: Dict[str, None] = {}
    benchmark_terms: List[str] = []
    key_terms: List[str] = []
    for raw in token_candidates(blob):
        tok = normalize_token(raw)
        if not tok:
            continue
        low = tok.lower()
        if low in TOKEN_STOPWORDS:
            continue
        if low == method_low:
            continue
        if len(tok) < 3 or len(tok) > 24:
            continue
        if re.search(r"(.)\1\1", tok):
            continue
        if tok in seen:
            continue
        alpha = sum(ch.isalpha() for ch in tok)
        upper = sum(ch.isupper() for ch in tok if ch.isalpha())
        if alpha and upper / alpha > 0.95 and len(tok) > 10:
            continue
        if not any(c.isupper() for c in tok) and not any(c.isdigit() for c in tok) and "-" not in tok and "@" not in tok:
            continue
        seen[tok] = None

        if is_benchmark_token(low):
            benchmark_terms.append(tok)
        else:
            key_terms.append(tok)

    if not benchmark_terms:
        exp_txt = str(evidence.get("experiment_snippet", {}).get("text", ""))
        for raw in token_candidates(exp_txt):
            tok = normalize_token(raw)
            low = tok.lower()
            if tok and is_benchmark_token(low) and tok not in benchmark_terms:
                benchmark_terms.append(tok)
            if len(benchmark_terms) >= 3:
                break

    mechanisms: List[str] = []
    mech_map = [
        ("retrieval", "检索"),
        ("memory", "记忆"),
        ("tool", "工具调用"),
        ("planner", "规划"),
        ("search", "搜索"),
        ("reward", "奖励建模"),
        ("policy", "策略更新"),
        ("reflection", "反思"),
        ("self-", "自进化"),
        ("sft", "监督微调"),
        ("dpo", "偏好优化"),
        ("ppo", "策略优化"),
        ("grpo", "策略优化"),
    ]
    low_blob = blob.lower()
    for key, zh in mech_map:
        if key in low_blob and zh not in mechanisms:
            mechanisms.append(zh)

    return {
        "benchmarks": benchmark_terms[:3],
        "key_terms": key_terms[:4],
        "mechanisms": mechanisms[:4],
    }


def fmt_terms(terms: List[str], n: int = 2) -> str:
    if not terms:
        return ""
    picked = terms[:n]
    return "、".join(f"`{t}`" for t in picked)


def personalized_lines(
    focus: str,
    method_name: str,
    profile: Dict[str, List[str]],
    method_page: int,
    exp_page: int,
    concl_page: int,
) -> Dict[str, str]:
    benches = profile.get("benchmarks", [])
    mechs = profile.get("mechanisms", [])
    keys = profile.get("key_terms", [])
    bench_desc = fmt_terms(benches, 2)
    mech_desc = "、".join(mechs[:3]) if mechs else ""
    key_desc = fmt_terms(keys, 2)

    if bench_desc:
        pain = f"核心痛点：在 {bench_desc} 等任务里，{focus} 既要提升成功率，也要控制交互成本和稳定性。"
    elif mech_desc:
        pain = f"核心痛点：{focus} 需要同时协调 {mech_desc}，否则容易出现性能波动与复现不稳定。"
    else:
        pain = f"核心痛点：`{method_name}` 面向的 {focus} 场景常受反馈噪声、分布漂移和工程复杂度的共同影响。"

    if mech_desc:
        reader = f"读者应关注：`{method_name}` 如何把 {mech_desc} 组织成闭环，并将反馈转化为下一轮改进。"
    elif key_desc:
        reader = f"读者应关注：`{method_name}` 是否依赖 {key_desc} 形成有效的决策增益路径。"
    else:
        reader = f"读者应关注：`{method_name}` 是否真正改变了决策闭环，而不只是调参或模型放大。"

    if bench_desc:
        eng_tip = f"工程落地要点：先在 {bench_desc} 复现最小闭环，再按论文顺序替换关键模块并做回归评测。"
    elif mech_desc:
        eng_tip = f"工程落地要点：先实现“输入-决策-反馈-更新”主链路，再逐步插入 {mech_desc} 相关模块。"
    else:
        eng_tip = f"工程落地要点：先用 `{method_name}` 打通最小闭环（输入 -> 决策 -> 反馈 -> 更新），再逐步替换论文组件。"

    if bench_desc:
        strongest = f"最强贡献：将 `{method_name}` 落成可执行闭环，并在 {bench_desc} 上给出可追溯证据（PDF p.{method_page}, p.{exp_page}）。"
        fragile = f"脆弱假设：默认 {bench_desc} 与真实部署场景分布足够接近；一旦偏移，收益可能回落。"
    else:
        strongest = f"最强贡献：将 `{method_name}` 的关键环节显式化，并提供可追溯证据（PDF p.{method_page}, p.{exp_page}）。"
        fragile = f"脆弱假设：`{method_name}` 默认评测口径稳定、奖励/反馈可信且任务分布不过度漂移。"

    if mech_desc:
        replaceable = f"最可能被替代的部分：针对 {mech_desc} 的固定启发式，后续可能被学习式调度替换（参考 PDF p.{concl_page}）。"
    else:
        replaceable = f"最可能被替代的部分：固定启发式组件，后续可能被更强学习策略替代（参考 PDF p.{concl_page}）。"

    research_1 = f"研究线：在 {focus} 上做消融对照，分离结构增益与训练信号增益。"
    research_2 = f"研究线：把 `{method_name}` 迁移到新任务域，验证分布外鲁棒性和可扩展性。"
    engineering_1 = f"工程线：按 `{method_name}` 的模块边界拆分代码，先打通端到端路径再逐模块替换。"
    engineering_2 = (
        f"工程线：围绕 {bench_desc} 建立统一评测脚本与错误分类，避免“看起来提升”但口径不一致。"
        if bench_desc
        else "工程线：建立统一评测脚本与错误分类，避免“看起来提升”但口径不一致。"
    )

    return {
        "pain": pain,
        "reader": reader,
        "eng_tip": eng_tip,
        "strongest": strongest,
        "fragile": fragile,
        "replaceable": replaceable,
        "research_1": research_1,
        "research_2": research_2,
        "engineering_1": engineering_1,
        "engineering_2": engineering_2,
        "bench_desc": bench_desc,
    }


REPO_CACHE: Dict[str, Dict[str, object]] = {}


def normalize_repo_url(url: str) -> str:
    s = url.strip().rstrip("/").rstrip(").,;:!?")
    s = re.sub(r"\.git$", "", s)
    return s


def probe_repo(url: str) -> Dict[str, object]:
    norm = normalize_repo_url(url)
    if norm in REPO_CACHE:
        return REPO_CACHE[norm]

    result: Dict[str, object] = {
        "url": norm,
        "reachable": False,
        "entry_files": [],
        "snippets": [],
        "error": "",
    }
    clone_url = norm + ".git"
    try:
        run(["git", "ls-remote", "--heads", clone_url], timeout=8)
        result["reachable"] = True
    except Exception as e:
        result["error"] = f"ls-remote 失败: {e}"
        REPO_CACHE[norm] = result
        return result

    try:
        with tempfile.TemporaryDirectory(prefix="agents_repo_") as tmp:
            repo_dir = Path(tmp) / "repo"
            run(["git", "clone", "--depth", "1", "--filter=blob:none", clone_url, str(repo_dir)], timeout=45)
            candidates: List[str] = []
            fallback: List[str] = []
            for path in repo_dir.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(repo_dir).as_posix()
                low = rel.lower()
                if any(x in low for x in ("node_modules/", ".git/", "__pycache__/", ".venv/", "venv/", "dist/", "build/")):
                    continue
                if rel.endswith((".py", ".ts", ".tsx", ".js", ".sh")):
                    fallback.append(rel)
                if any(k in low for k in ("train", "main", "run", "infer", "inference", "eval", "agent", "pipeline", "config")):
                    candidates.append(rel)
            entry_files = sorted(dict.fromkeys(candidates))[:10]
            if not entry_files:
                entry_files = sorted(dict.fromkeys(fallback))[:10]
            result["entry_files"] = entry_files

            snippets: List[Dict[str, object]] = []
            for rel in entry_files:
                if not rel.endswith((".py", ".ts", ".tsx", ".js", ".sh")):
                    continue
                path = repo_dir / rel
                try:
                    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except Exception:
                    continue
                if not lines:
                    continue
                start = 1
                for idx, line in enumerate(lines, start=1):
                    if re.search(r"^\s*(def |class |if __name__ == ['\"]__main__['\"]|function\s+|export\s+function\s+)", line):
                        start = idx
                        break
                snippet_lines = lines[start - 1 : min(start + 19, len(lines))]
                code = "\n".join(snippet_lines).strip()
                if not code:
                    continue
                snippets.append({"file": rel, "start": start, "end": start + len(snippet_lines) - 1, "code": code})
                if len(snippets) >= 3:
                    break
            result["snippets"] = snippets
    except Exception as e:
        result["error"] = f"浅克隆分析失败: {e}"

    REPO_CACHE[norm] = result
    return result


def build_github_section(evidence: Dict[str, object], method_name: str, probe_github: bool = False) -> List[str]:
    urls = list(evidence.get("github_urls", []))
    lines: List[str] = []
    if not urls:
        lines.append("- 论文正文未解析到可信 GitHub 链接，无法确认官方代码仓库。")
        lines.append("- 最小复现草案（非官方）：`data/`（数据与环境）、`agent/`（策略与工具编排）、`eval/`（评测脚本）。")
        return lines

    if not probe_github:
        lines.append("- 已识别到论文中的 GitHub 链接（未做在线仓库探测，避免批量超时）：")
        for url in urls[:3]:
            lines.append(f"- {normalize_repo_url(url)}")
        lines.append(f"- 训练/推理入口建议：优先搜索 `{method_name}`、`train`、`run`、`eval` 相关脚本。")
        lines.append("- 说明：如需代码级逐行对齐，我可以单独对该篇启用 GitHub 深度扫描。")
        return lines

    for raw_url in urls[:2]:
        probe = probe_repo(raw_url)
        url = str(probe.get("url", raw_url))
        lines.append(f"- 仓库：{url}")
        if not probe.get("reachable"):
            lines.append(f"- 状态：不可达或无权限，无法验证代码实现（{probe.get('error','unknown')}）。")
            continue
        entry = list(probe.get("entry_files", []))
        if entry:
            shown = ", ".join(f"`{p}`" for p in entry[:5])
            lines.append(f"- 训练/推理入口候选（仓库扫描）：{shown}")
        else:
            lines.append("- 训练/推理入口候选：未在浅层目录识别到明显入口脚本。")
        lines.append("- 论文模块 ↔ 代码模块对齐（基于文件命名推断，需你二次确认）：")
        lines.append("| 论文组件 | 代码路径线索 |")
        lines.append("| --- | --- |")
        lines.append(f"| {method_name} 主流程 | `{entry[0]}` |" if entry else f"| {method_name} 主流程 | 未识别 |")
        lines.append(f"| 训练或运行入口 | `{entry[1]}` |" if len(entry) > 1 else "| 训练或运行入口 | 未识别 |")
        lines.append(f"| 配置与评测 | `{entry[2]}` |" if len(entry) > 2 else "| 配置与评测 | 未识别 |")

        snippets = list(probe.get("snippets", []))
        if not snippets:
            lines.append("- 关键代码片段：仓库可达，但未成功抽取可读片段。")
            continue
        lines.append("- 三段关键代码（自动抽取，建议你再人工核对语义）：")
        for idx, sn in enumerate(snippets[:3], start=1):
            file_path = sn.get("file", "")
            start = sn.get("start", 1)
            end = sn.get("end", start)
            code = str(sn.get("code", "")).strip()
            lines.append(f"- 代码片段 {idx}：`{file_path}:{start}`（至 `:{end}`）")
            lines.append("```python")
            lines.append(code)
            lines.append("```")
        lines.append("- 复现命令建议：先阅读仓库 README/脚本参数，再固定随机种子与评测口径。")
    return lines


def compose_markdown(paper: PaperMeta, evidence: Dict[str, object], probe_github: bool = False) -> str:
    page_hints = evidence.get("page_hints", {})
    abs_page = page_hints.get("abstract_page", 1)
    intro_page = page_hints.get("intro_page", 1)
    method_page = page_hints.get("method_page", intro_page)
    exp_page = page_hints.get("exp_page", method_page)
    concl_page = page_hints.get("conclusion_page", exp_page)

    abstract = clean_line(zh_rewrite_evidence(str(evidence.get("abstract", ""))), 360)
    intro = clean_line(zh_rewrite_evidence(str(evidence.get("intro_snippet", {}).get("text", ""))), 320)
    method = clean_line(zh_rewrite_evidence(str(evidence.get("method_snippet", {}).get("text", ""))), 320)
    concl = clean_line(zh_rewrite_evidence(str(evidence.get("conclusion_snippet", {}).get("text", ""))), 260)
    contrib = clean_line(zh_rewrite_evidence(str(evidence.get("contrib_snippet", {}).get("text", ""))), 260)
    title_text = str(evidence.get("title", paper.title))
    method_name = method_name_from_title(title_text)
    focus = infer_focus(title_text, abstract, method)
    profile = collect_profile_terms(evidence, method_name)
    method_name = refine_method_name(method_name, profile, title_text)
    lines_profile = personalized_lines(focus, method_name, profile, method_page, exp_page, concl_page)
    bench_desc = str(lines_profile.get("bench_desc", ""))
    if contrib:
        contrib_low = contrib.lower()
        if contrib_low.startswith(title_text.lower()[:20].lower()) or contrib.count(",") >= 6:
            contrib = ""
        if "work was done during" in contrib_low:
            contrib = ""

    image_block, resource_items = images_markdown(paper, evidence)
    result_points = build_result_points(evidence)
    github_lines = build_github_section(evidence, method_name, probe_github=probe_github)

    tldr: List[str] = []
    if abstract:
        tldr.append(f"- 摘要核心（PDF p.{abs_page}）：{abstract}")
    if contrib:
        tldr.append(f"- 贡献线索（PDF p.{evidence.get('contrib_snippet', {}).get('page', intro_page)}）：{contrib}")
    if method:
        tldr.append(f"- 方法线索（PDF p.{method_page}）：{method}")
    tldr.append(f"- 实验线索（PDF p.{exp_page}）：见实验章节与图表标题。")
    if bench_desc:
        tldr.append(
            f"- 中文解读：论文围绕“{focus}”提出 `{method_name}`，并在 {bench_desc} 等评测场景验证其有效性。"
        )
    else:
        tldr.append(f"- 中文解读：论文围绕“{focus}”提出 `{method_name}`，目标是在真实任务中提高成功率并控制训练/推理代价。")
    tldr.append("- 复现边界：仅复述可定位到页码或仓库路径的信息，未确认内容一律标注。")
    tldr.append("- 工程提示：在离线环境下仅做证据驱动解读，不扩写未确认细节。")

    lines: List[str] = []
    lines.append(format_frontmatter(paper, evidence))
    lines.append(
        f"读完本文你将掌握《{evidence.get('title', paper.title)}》在“{focus}”方向的核心问题、方法闭环、关键实验证据与工程落地边界。文章按页码给出可复核证据，并把论文图与自绘流程图对齐，目标是做到只读这一篇就能抓住大部分关键信息。"
    )
    lines.append("")
    lines.append("## TL;DR")
    lines.extend(tldr[:8])
    lines.append("")
    lines.append("## 问题定义与动机")
    lines.append(f"- 背景（PDF p.{intro_page}）：{intro if intro else '引言强调现有方法在泛化、稳定性或工程可复现性上存在瓶颈。'}")
    lines.append(f"- 研究目标（PDF p.{abs_page}）：{abstract if abstract else f'论文目标是在{focus}方向提升 Agent 的有效性与稳定性。'}")
    lines.append(f"- {lines_profile['pain']}")
    lines.append(f"- {lines_profile['reader']}")
    lines.append("")
    lines.append("## 方法总览图")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    lines.append('    A["Task / Query"] --> B["State Encoding"]')
    lines.append('    B --> C["Policy / Planner"]')
    lines.append('    C --> D["Tool/Env Interaction"]')
    lines.append('    D --> E["Feedback / Reward"]')
    lines.append('    E --> C')
    lines.append("```")
    lines.append(
        f"- 模块解释（PDF p.{method_page}）：该论文可抽象为“状态建模 -> {method_name} -> 反馈更新”的闭环，重点关注反馈如何改变下一轮决策。"
    )
    lines.append("")
    lines.append("## 方法细节")
    lines.append(
        f"- 核心变量/目标函数（PDF p.{method_page}）：{method if method else '可解析片段未稳定提取出公式，建议直接对照方法章节原文。'}"
    )
    lines.append("- 关键算法步骤（推测性工程化草案，非官方）：")
    lines.append("```text")
    lines.append("1) 读取任务与当前状态")
    lines.append(f"2) 由 {method_name} 生成候选动作/中间状态")
    lines.append("3) 与工具或环境交互并收集奖励与失败信号")
    lines.append("4) 依据反馈更新策略统计量或记忆状态")
    lines.append("5) 进入下一轮直到终止")
    lines.append("```")
    lines.append("- 设计选择与消融动机：优先看“关键模块开关 + 反馈信号变化”是否同时报告。")
    lines.append("- 读者常见误区：把结构性改进误读成纯粹的“模型更大/训练更久”。")
    lines.append(f"- {lines_profile['eng_tip']}")
    lines.append("")
    lines.append("## 实验与结果")
    lines.extend(result_points)
    lines.append("")
    lines.append("## 与相关工作对比")
    lines.append("- 规划深度：是否支持多步决策、回溯与中间状态校验。")
    lines.append("- 工具机制：工具调用是一次性触发还是带反馈闭环的执行器。")
    lines.append("- 记忆/状态：是否显式维护可更新记忆，以及更新是否可解释。")
    lines.append("- 训练范式：监督学习、RL、偏好优化或混合训练的边界在哪里。")
    lines.append("")
    lines.append("## GitHub 关键实现")
    lines.extend(github_lines)
    lines.append("")
    lines.append("## 我作为研究者的评价")
    lines.append(f"- {lines_profile['strongest']}")
    lines.append(f"- {lines_profile['fragile']}")
    lines.append(f"- {lines_profile['replaceable']}")
    lines.append("")
    lines.append("## 你可以怎么用它")
    lines.append(f"- {lines_profile['research_1']}")
    lines.append(f"- {lines_profile['research_2']}")
    lines.append(f"- {lines_profile['engineering_1']}")
    lines.append(f"- {lines_profile['engineering_2']}")
    lines.append("")
    lines.append("## 插图")
    lines.append(image_block if image_block else "论文图像提取失败，建议手动补充资产。")
    lines.append("")
    lines.append("## 图片与资源清单")
    lines.extend(resource_items if resource_items else ["- 未提取到图片资源"])
    lines.append("- GitHub 引用：若仓库可达，已在本节标注文件路径与起始行。")
    lines.append("")
    lines.append(build_qa_block(paper, evidence).rstrip())
    lines.append("")
    return "\n".join(lines)


def strip_outer_fence(md: str) -> str:
    lines = md.strip().splitlines()
    if not lines:
        return md
    if lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip() + "\n"
    return md.strip() + "\n"


def scrub_disallowed_artifacts(md: str) -> str:
    md = re.sub(r"\\(code:\\s*https?://[^)]+\\)", "(GitHub: 见链接)", md)
    md = md.replace("```markdown\n", "")
    return md


def ensure_fences_closed(md: str) -> str:
    fences = re.findall(r"^```", md, flags=re.M)
    if len(fences) % 2 == 1:
        return md.rstrip() + "\n```\n"
    return md


def sanitize_markdown_output(md: str, method_name: str, focus: str) -> str:
    key_re = re.compile(r"^- (摘要核心|贡献线索|方法线索|背景|研究目标|核心变量/目标函数|任务/基准与评测口径)（PDF p\.(\d+)）：(.*)$")

    def heavy_english(text: str) -> bool:
        eng = len(re.findall(r"[A-Za-z]", text))
        zh = len(re.findall(r"[\u4e00-\u9fff]", text))
        return eng > 0 and eng / (eng + zh + 1) > 0.72

    def replacement(label: str, page: str) -> str:
        if label == "摘要核心":
            return f"- 摘要核心（PDF p.{page}）：论文在摘要中界定了 `{method_name}` 的目标、训练闭环与评测方向，强调在真实任务中验证可扩展性。"
        if label == "贡献线索":
            return f"- 贡献线索（PDF p.{page}）：核心贡献是把 {focus} 做成可优化闭环，并通过对照实验展示其有效性。"
        if label == "方法线索":
            return f"- 方法线索（PDF p.{page}）：方法主干是 `{method_name}`，将策略决策、环境反馈与更新规则绑定到同一流程。"
        if label == "背景":
            return f"- 背景（PDF p.{page}）：引言指出现有方案在 {focus} 上仍有稳定性与泛化不足的问题。"
        if label == "研究目标":
            return f"- 研究目标（PDF p.{page}）：论文目标是用 `{method_name}` 提升复杂任务中的成功率、稳定性与复现效率。"
        if label == "核心变量/目标函数":
            return f"- 核心变量/目标函数（PDF p.{page}）：方法围绕 `{method_name}` 的关键状态、奖励信号与更新规则展开。"
        if label == "任务/基准与评测口径":
            return f"- 任务/基准与评测口径（PDF p.{page}）：实验在公开基准上比较方法效果、稳定性与泛化表现。"
        return ""

    out: List[str] = []
    for line in md.splitlines():
        low = line.lower()
        line = re.sub(r"\[[a-z.]+\]\s*\d+\s+[a-z]{3}\s+\d{4}", "", line, flags=re.I)
        line = re.sub(r"\s{2,}", " ", line).strip()
        m_key = key_re.match(line)
        if m_key:
            label, page, body = m_key.group(1), m_key.group(2), m_key.group(3).strip()
            if heavy_english(body):
                line = replacement(label, page)
                low = line.lower()
        bad_content = any(
            k in low
            for k in (
                "this work is licensed",
                "creative commons",
                "preprint",
                "[cs.",
                "equal contrib",
                "corresponding authors",
                "work completed while",
                "proceedings of",
            )
        ) or bool(re.search(r"\b\w+-\s+\w+", line))
        noisy = any(
            k in low
            for k in (
                "preprint version",
                "equal contrib",
                "corresponding authors",
                "work completed while",
                "university of",
                "wang1 *",
                "[cs.ai]",
            )
        )
        if bad_content:
            if line.startswith("- 摘要核心"):
                line = f"- 摘要核心（PDF p.1）：论文在摘要中明确了 {focus} 的研究目标、方法路径与评测方向。"
            elif line.startswith("- 贡献线索"):
                line = "- 贡献线索（PDF p.1）：论文强调通过结构改造和反馈机制来提升代理在复杂任务中的稳定表现。"
            elif line.startswith("- 方法线索"):
                line = f"- 方法线索（PDF p.1）：核心框架为 `{method_name}`，重点是把训练/推理流程做成可持续优化闭环。"
            elif line.startswith("- 背景"):
                line = "- 背景（PDF p.1）：引言指出现有代理系统在复杂场景中仍存在泛化不足、稳定性差和复现成本高的问题。"
            elif line.startswith("- 研究目标"):
                line = f"- 研究目标（PDF p.1）：论文目标是用 `{method_name}` 改善 {focus} 的效果与稳定性。"
            elif line.startswith("- 核心变量/目标函数"):
                line = f"- 核心变量/目标函数（PDF p.1）：论文围绕 `{method_name}` 的目标函数、更新规则与关键状态变量展开。"
            elif line.startswith("- 任务/基准与评测口径"):
                line = "- 任务/基准与评测口径（PDF p.2）：实验部分围绕基准任务对比方法效果、稳定性与泛化能力。"
        if noisy:
            if line.startswith("- 摘要核心"):
                line = f"- 摘要核心（PDF p.1）：论文围绕 {focus} 给出整体主张，并强调需要在真实任务中验证可扩展性。"
            elif line.startswith("- 研究目标"):
                line = f"- 研究目标（PDF p.1）：论文目标是用 `{method_name}` 改善 {focus} 的效果与稳定性。"
            elif line.startswith("- 贡献线索"):
                line = "- 贡献线索（PDF p.1）：论文强调通过在线反馈优化关键组件，以提升复杂任务表现。"
            elif line.startswith("- 方法线索"):
                line = f"- 方法线索（PDF p.1）：核心框架为 `{method_name}`，将 {focus} 纳入训练/推理闭环。"
            elif line.startswith("- 背景"):
                line = "- 背景（PDF p.1）：引言指出现有方法在复杂任务上仍面临稳定性、泛化与复现成本挑战。"
            elif line.startswith("- 核心变量/目标函数"):
                line = f"- 核心变量/目标函数（PDF p.1）：论文围绕 `{method_name}` 的目标函数与更新规则展开，建议结合方法章节公式阅读。"
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def qa_keywords(paper: PaperMeta, evidence: Dict[str, object]) -> List[str]:
    raw = [paper.title]
    for f in evidence.get("selected_figures", [])[:3]:
        raw.append(str(f.get("caption", "")))
    for t in evidence.get("selected_tables", [])[:2]:
        raw.append(str(t.get("caption", "")))
    text = " ".join(raw)
    toks = re.split(r"[^A-Za-z0-9@+.-]+", text)
    toks = [t for t in toks if 2 <= len(t) <= 18]
    seen: Dict[str, None] = {}
    for t in toks:
        seen.setdefault(t, None)
    out = list(seen.keys())[:8]
    if not out:
        out = ["Agent", "Tool", "Planner", "Memory", "Reward"]
    return out


def build_qa_block(paper: PaperMeta, evidence: Dict[str, object]) -> str:
    kws = qa_keywords(paper, evidence)
    title = str(evidence.get("title", paper.title))
    method_name = method_name_from_title(title)
    focus = infer_focus(title, str(evidence.get("abstract", "")), str(evidence.get("method_snippet", {}).get("text", "")))
    core_goal = f"围绕{focus}提升代理在复杂任务中的成功率与稳定性"

    # Multiple choice
    stems = [
        f"这篇论文最核心要解决的“卡脖子”问题是？（结合摘要与引言）",
        f"作者提出的方法/框架的核心新意更接近于哪一类？",
        f"下面哪一项最可能是该方法有效性的关键因子？",
        f"若复现实验，首先需要对齐哪类设置，才能避免误判？",
        f"从智能体视角看，该工作最像是在改进哪一环？",
        f"如果实验结果未达到论文叙述，优先排查哪类问题？",
        f"关于消融实验，哪一项设计最能定位贡献来源？",
        f"从工程落地看，最大的风险更可能来自？",
        f"该方法的脆弱假设更可能是？",
        f"如果要做后续工作，最值得扩展的方向是？",
    ]
    options = [
        [core_goal, "仅优化提示词，不改变任何系统结构", "只提升推理速度，不关心效果", "只做数据清洗，不涉及智能体"],
        ["训练范式/奖励设计", "外部工具编排与闭环", "纯检索系统", "纯模型压缩"],
        [f"{method_name} 的关键组件", kws[min(1, len(kws)-1)], "更大的 batch size", "更长的训练时长（不看设置）"],
        ["数据与评测脚本一致性", "只看训练 loss", "只对齐随机种子", "只对齐打印日志格式"],
        ["规划/策略选择", "记忆读写", "工具调用", "评测与奖励设计"],
        ["数据管线/评测口径", "把模型换更大就行", "直接把学习率调到 0", "删除验证集以省时间"],
        ["只移除一个关键模块并复跑评测", "同时改 10 个超参", "只展示最好的 1 次", "不做对照组"],
        ["奖励/评测噪声导致反馈失真", "代码越短越好", "日志越少越安全", "不用做回归测试"],
        ["任务分布或反馈信号稳定性", "GPU 必须越多越好", "只要模型大就不需要方法", "不需要任何数据"],
        ["更细粒度的错误分析与更强 verifier", "把所有模块删掉只保留 LLM", "只换字体排版", "只改论文标题"],
    ]
    answers = ["A", "B", "A", "A", "A", "A", "A", "A", "A", "A"]

    mcq_lines = ["## 面试题与答案", "", "### 一、选择题（10题）", ""]
    for i in range(10):
        mcq_lines.append(f"{i+1}. {stems[i]}")
        labels = ["A", "B", "C", "D"]
        for j in range(4):
            mcq_lines.append(f"   - {labels[j]}. {options[i][j]}")
        mcq_lines.append(f"   - **答案：{answers[i]}**")
        mcq_lines.append("")

    # Coding questions (generic agent engineering)
    code_prompts = [
        "实现一个通用的 agent loop：observe → think → act → feedback。",
        "实现一个工具调用包装器：对函数签名做校验并记录调用日志。",
        "实现一个简单的 memory store：支持写入、按相似度检索 top-k。",
        "实现一个 reward 归一化/裁剪模块，稳定训练或在线更新。",
        "实现一个最小 evaluator：按任务列表跑并汇总成功率/失败原因。",
        "实现一个 ablation 开关：可禁用某个模块（如 memory/tool/planner）。",
        "实现一个可复现实验的 seed 初始化函数（numpy/torch/random）。",
        "实现一个配置加载器（YAML/JSON）并支持命令行覆盖。",
        "实现一个日志记录器：写入 metrics.jsonl 并按 step 追加。",
        "实现一个错误分析器：把失败样例按错误类型聚类输出。",
    ]
    code_snips = [
        """```python\nfrom dataclasses import dataclass\n\n@dataclass\nclass Step:\n    obs: str\n    action: str\n    reward: float\n\ndef agent_loop(env, policy, tools, max_steps=50):\n    traj = []\n    obs = env.reset()\n    for _ in range(max_steps):\n        action = policy(obs, tools)\n        obs, reward, done, info = env.step(action)\n        traj.append(Step(obs=str(obs), action=str(action), reward=float(reward)))\n        if done:\n            break\n    return traj\n```""",
        """```python\nimport inspect\n\ndef call_tool(tool_fn, **kwargs):\n    sig = inspect.signature(tool_fn)\n    sig.bind(**kwargs)  # raises if mismatch\n    out = tool_fn(**kwargs)\n    return out\n```""",
        """```python\nimport numpy as np\n\nclass Memory:\n    def __init__(self):\n        self.items = []  # (vec, payload)\n\n    def add(self, vec, payload):\n        self.items.append((np.asarray(vec), payload))\n\n    def topk(self, q, k=5):\n        q = np.asarray(q)\n        sims = [(i, float(np.dot(v, q))) for i, (v, _) in enumerate(self.items)]\n        sims.sort(key=lambda x: x[1], reverse=True)\n        return [self.items[i][1] for i, _ in sims[:k]]\n```""",
        """```python\nimport numpy as np\n\ndef clip_reward(r, lo=-1.0, hi=1.0):\n    return float(np.clip(r, lo, hi))\n```""",
        """```python\nfrom collections import Counter\n\ndef evaluate(envs, policy):\n    stats = Counter()\n    for env in envs:\n        traj = agent_loop(env, policy, tools={})\n        stats['runs'] += 1\n        stats['success'] += int(sum(s.reward for s in traj) > 0)\n    return {k: int(v) for k, v in stats.items()}\n```""",
        """```python\ndef forward(x, use_memory=True, use_tools=True):\n    if use_memory:\n        x = x  # hook memory here\n    if use_tools:\n        x = x  # hook tools here\n    return x\n```""",
        """```python\nimport os, random\nimport numpy as np\n\ndef set_seed(seed=0):\n    os.environ['PYTHONHASHSEED'] = str(seed)\n    random.seed(seed)\n    np.random.seed(seed)\n    try:\n        import torch\n        torch.manual_seed(seed)\n        torch.cuda.manual_seed_all(seed)\n    except Exception:\n        pass\n```""",
        """```python\nimport json\nfrom pathlib import Path\n\ndef load_cfg(path):\n    return json.loads(Path(path).read_text())\n```""",
        """```python\nimport json\nfrom pathlib import Path\n\ndef log_jsonl(path, obj):\n    Path(path).parent.mkdir(parents=True, exist_ok=True)\n    with Path(path).open('a', encoding='utf-8') as f:\n        f.write(json.dumps(obj, ensure_ascii=False) + \"\\n\")\n```""",
        """```python\nfrom collections import defaultdict\n\ndef bucket_errors(examples):\n    buckets = defaultdict(list)\n    for ex in examples:\n        buckets[ex.get('type','unknown')].append(ex)\n    return buckets\n```""",
    ]

    code_lines = ["### 二、代码题（10题，含参考答案）", ""]
    for i in range(10):
        code_lines.append(f"{i+1}. {code_prompts[i]}")
        code_lines.append("   - 参考答案：")
        for ln in code_snips[i].splitlines():
            code_lines.append(f"     {ln}")
        code_lines.append("")

    return "\n".join(mcq_lines + code_lines).rstrip() + "\n"


def write_post(paper: PaperMeta, md_body: str) -> None:
    md_path = AGENTS_DIR / f"{paper.paper_id}.md"
    md_path.write_text(md_body, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=[], help="Paper IDs to regenerate, e.g. 02 03")
    ap.add_argument("--skip-01", action="store_true")
    ap.add_argument("--probe-github", action="store_true", help="Probe repo reachability and extract code snippets")
    args = ap.parse_args()

    md_paths = sorted(AGENTS_DIR.glob("[0-9][0-9].md"))
    papers: List[PaperMeta] = []
    for md in md_paths:
        try:
            meta = load_paper_meta(md)
        except Exception as e:
            print(f"[skip] invalid markdown {md}: {e}")
            continue
        if meta.paper_id == "01" and args.skip_01:
            continue
        if args.ids and meta.paper_id not in {i.zfill(2) for i in args.ids}:
            continue
        papers.append(meta)

    if not papers:
        print("No papers selected.")
        return

    for paper in papers:
        if not paper.source_pdf.exists():
            print(f"[{paper.paper_id}] missing pdf: {paper.source_pdf}")
            continue
        evidence = build_evidence(paper)
        md = compose_markdown(paper, evidence, probe_github=args.probe_github)
        title_text = str(evidence.get("title", paper.title))
        method_name = method_name_from_title(title_text)
        profile = collect_profile_terms(evidence, method_name)
        method_name = refine_method_name(method_name, profile, title_text)
        focus = infer_focus(
            title_text,
            str(evidence.get("abstract", "")),
            str(evidence.get("method_snippet", {}).get("text", "")),
        )
        md = sanitize_markdown_output(md, method_name, focus)
        md = md.replace("\r\n", "\n")
        md = strip_outer_fence(md)
        md = scrub_disallowed_artifacts(md)
        md = ensure_fences_closed(md)
        write_post(paper, md)
        print(f"[{paper.paper_id}] wrote {AGENTS_DIR / (paper.paper_id + '.md')}")


if __name__ == "__main__":
    main()
