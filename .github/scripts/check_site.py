"""Validate the compact static CoupFE Core website and retained site evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = ROOT / "site"
INDEX = SITE_ROOT / "index.html"
STYLES = SITE_ROOT / "styles.css"
EVIDENCE = SITE_ROOT / "evidence.json"
REPOSITORY = "https://github.com/tengzhang48/CoupFE"
HERTZ_SITE_FIGURE = "hertz-contact-benchmark.svg"
LINEAR_BAR_SITE_FIGURE = "linear-bar-snapshot.svg"
NEO_HOOKEAN_BLOCK_SITE_FIGURE = "neo-hookean-block-snapshot.svg"
CONTACT_3D_BLOCKS_SITE_FIGURE = "contact-3d-blocks-snapshot.svg"
EXPECTED_SITE_FILES = {
    "evidence.json",
    HERTZ_SITE_FIGURE,
    LINEAR_BAR_SITE_FIGURE,
    NEO_HOOKEAN_BLOCK_SITE_FIGURE,
    CONTACT_3D_BLOCKS_SITE_FIGURE,
    "index.html",
    "styles.css",
}
EXPECTED_SECTIONS = ["top", "core", "backends", "evidence", "snapshots", "scope", "start"]


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_lang: str | None = None
        self.title_parts: list[str] = []
        self.in_title = False
        self.meta: dict[str, str] = {}
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.headings: list[tuple[int, str]] = []
        self._heading_level: int | None = None
        self._heading_parts: list[str] = []
        self.section_ids: list[str] = []
        self.landmarks: list[str] = []
        self.nav_labels: list[str] = []
        self.script_count = 0
        self.image_count = 0
        self.svg_image_count = 0
        self.image_sources: list[str] = []
        self.image_alts: list[str] = []

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {name: value or "" for name, value in attrs_list}
        if tag == "html":
            self.html_lang = attrs.get("lang")
        if tag == "title":
            self.in_title = True
        if tag == "meta" and attrs.get("name"):
            self.meta[attrs["name"]] = attrs.get("content", "")
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "a" and attrs.get("href"):
            self.hrefs.append(attrs["href"])
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._heading_parts = []
        if tag == "section" and attrs.get("id"):
            self.section_ids.append(attrs["id"])
        if tag in {"header", "main", "footer"}:
            self.landmarks.append(tag)
        if tag == "nav":
            self.landmarks.append(tag)
            self.nav_labels.append(attrs.get("aria-label", ""))
        if tag == "script":
            self.script_count += 1
        if tag == "img":
            self.image_count += 1
            self.image_sources.append(attrs.get("src", ""))
            self.image_alts.append(attrs.get("alt", ""))
        if tag == "svg" and attrs.get("role") == "img":
            self.svg_image_count += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if self._heading_level is not None and tag == f"h{self._heading_level}":
            text = " ".join("".join(self._heading_parts).split())
            self.headings.append((self._heading_level, text))
            self._heading_level = None
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if self._heading_level is not None:
            self._heading_parts.append(data)


def fail(message: str) -> None:
    raise SystemExit(f"site check failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pinned_source_files(
    source_commit: str,
    source_files: dict[str, object],
) -> None:
    """Verify source hashes at the named commit when Git history is present."""

    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return
    if probe.returncode != 0:
        return

    for relative, expected_hash in source_files.items():
        retained = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(retained.returncode == 0, f"source commit is missing: {relative}")
        retained_hash = hashlib.sha256(retained.stdout).hexdigest()
        require(
            retained_hash == expected_hash,
            f"source hash is not reproducible at sourceCommit: {relative}",
        )


def check_evidence(payload: dict[str, object]) -> None:
    require(payload.get("schemaVersion") == 2, "unexpected evidence schema")
    require(payload.get("recordedDate") == "2026-08-10", "unexpected evidence date")
    source_commit = payload.get("sourceCommit")
    require(
        isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
        "source commit must be a full SHA",
    )
    source_scope = payload.get("sourceCommitScope")
    require(
        isinstance(source_scope, str)
        and "sourceFiles" in source_scope
        and "presentationArtifacts" in source_scope,
        "source-commit scope must distinguish sources from presentation artifacts",
    )

    source_files = payload.get("sourceFiles")
    require(isinstance(source_files, dict) and source_files, "source-file hashes are required")
    for relative, expected_hash in source_files.items():
        require(isinstance(relative, str), "source path must be text")
        require(isinstance(expected_hash, str), f"hash for {relative} must be text")
        path = ROOT / relative
        require(path.is_file(), f"evidence source is missing: {relative}")
        require(sha256(path) == expected_hash, f"evidence source changed: {relative}")
    check_pinned_source_files(source_commit, source_files)

    presentation = payload.get("presentationArtifacts")
    require(isinstance(presentation, dict) and presentation, "presentation-artifact record is required")
    require("hertzContactFigure" in presentation, "Hertz presentation artifact is required")
    for name, artifact in presentation.items():
        require(isinstance(artifact, dict), f"presentation artifact must be an object: {name}")
        renderer = artifact.get("renderer")
        renderer_hash = artifact.get("rendererSha256")
        require(isinstance(renderer, str), f"renderer path must be text: {name}")
        require(isinstance(renderer_hash, str), f"renderer hash must be text: {name}")
        require((ROOT / renderer).is_file(), f"renderer is missing: {name}")
        require(sha256(ROOT / renderer) == renderer_hash, f"renderer changed: {name}")
        require(
            source_files.get(renderer) == renderer_hash,
            f"renderer must be source-pinned: {name}",
        )
        artifact_files = artifact.get("files")
        require(isinstance(artifact_files, dict) and artifact_files, f"artifact files are required: {name}")
        require(
            not set(source_files).intersection(artifact_files),
            f"presentation artifacts must not be classified as source files: {name}",
        )
        for relative, expected_hash in artifact_files.items():
            require(isinstance(relative, str), f"artifact path must be text: {name}")
            require(isinstance(expected_hash, str), f"artifact hash for {relative} must be text: {name}")
            path = ROOT / relative
            require(path.is_file(), f"presentation artifact is missing: {relative}")
            require(sha256(path) == expected_hash, f"presentation artifact changed: {relative}")
    require(
        presentation["hertzContactFigure"].get("derivedFromRun") == "hertzContact",
        "Hertz artifact must name its source run",
    )

    runs = payload.get("runs")
    require(isinstance(runs, dict), "runs mapping is required")
    hertz = runs.get("hertzContact")
    require(isinstance(hertz, dict), "Hertz record is required")
    deltas = hertz.get("deltas")
    forces_fe = hertz.get("forcesFe")
    forces_hertz = hertz.get("forcesHertz")
    require(
        isinstance(deltas, list)
        and isinstance(forces_fe, list)
        and isinstance(forces_hertz, list)
        and len(deltas) == len(forces_fe) == len(forces_hertz) == 5,
        "Hertz arrays must contain five samples",
    )
    expected_forces = [
        (4.0 / 3.0) * (10.0 / (1.0 - 0.3**2)) * math.sqrt(2.0) * float(delta) ** 1.5
        for delta in deltas
    ]
    require(
        all(abs(float(actual) - expected) < 1.0e-12 for actual, expected in zip(forces_hertz, expected_forces)),
        "stored Hertz reference force does not match the stated law",
    )
    log_deltas = [math.log(float(value)) for value in deltas]
    log_forces = [math.log(float(value)) for value in forces_fe]
    mean_x = sum(log_deltas) / len(log_deltas)
    mean_y = sum(log_forces) / len(log_forces)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_deltas, log_forces)) / sum(
        (x - mean_x) ** 2 for x in log_deltas
    )
    require(abs(float(hertz.get("logLogSlope", 0.0)) - slope) < 5.0e-4, "stored Hertz slope drifted")
    ratios = [float(fe) / float(reference) for fe, reference in zip(forces_fe, forces_hertz)]
    require(round(min(ratios), 2) == hertz.get("forceRatioMin"), "Hertz minimum ratio drifted")
    require(round(max(ratios), 2) == hertz.get("forceRatioMax"), "Hertz maximum ratio drifted")
    force_errors = [(ratio - 1.0) * 100.0 for ratio in ratios]
    require(
        round(min(force_errors), 1) == hertz.get("forceErrorPercentMin"),
        "Hertz minimum force error drifted",
    )
    require(
        round(max(force_errors), 1) == hertz.get("forceErrorPercentMax"),
        "Hertz maximum force error drifted",
    )
    require(hertz.get("meshShape") == [16, 16, 8], "Hertz mesh shape drifted")
    require(
        hertz.get("nodes") == 2601
        and hertz.get("elements") == 2048
        and hertz.get("degreesOfFreedom") == 7803,
        "Hertz problem size drifted",
    )
    figure = hertz.get("figure")
    site_figure = hertz.get("siteFigure")
    require(
        isinstance(figure, str) and (ROOT / figure).is_file(),
        "Hertz solver-backed figure is missing",
    )
    require(
        isinstance(site_figure, str)
        and site_figure == f"site/{HERTZ_SITE_FIGURE}"
        and (ROOT / site_figure).is_file(),
        "Hertz website figure is missing",
    )
    hertz_figure_files = presentation["hertzContactFigure"]["files"]
    require(
        figure in hertz_figure_files and site_figure in hertz_figure_files,
        "Hertz figures must be presentation artifacts",
    )
    require(hertz.get("status") == "OK", "Hertz rerun did not pass")

    block = runs.get("neoHookeanBlock")
    require(isinstance(block, dict), "Neo-Hookean block record is required")
    require(float(block.get("reportedRelativeError", 1.0)) < 1.0e-12, "block rerun error is too large")

    annulus = runs.get("curvedAnnulus")
    require(isinstance(annulus, dict), "curved-annulus record is required")
    errors = annulus.get("reembeddedRelativeL2Errors")
    require(isinstance(errors, list) and len(errors) == 4, "annulus record must contain four levels")
    require(all(float(a) > float(b) for a, b in zip(errors, errors[1:])), "annulus errors must decrease")
    last_rate = math.log(float(errors[-2]) / float(errors[-1]), 2.0)
    require(round(last_rate, 2) == annulus.get("lastObservedRate"), "annulus rate drifted")

    collision = runs.get("contact3dBlocks")
    friction = runs.get("contact3dFriction")
    require(isinstance(collision, dict) and isinstance(friction, dict), "3-D contact records are required")
    require(
        collision.get("collisionObserved") is True
        and collision.get("penetrationFree") is True
        and float(collision.get("minimumInterfaceGap", -1.0)) > 0.0
        and collision.get("status") == "OK",
        "3-D collision rerun did not pass",
    )
    require(
        friction.get("held") is True
        and friction.get("penetrationFree") is True
        and float(friction.get("slipRatio", 1.0)) < 0.70
        and float(friction.get("minimumInterfaceGap", -1.0)) > 0.0
        and friction.get("status") == "OK",
        "3-D friction rerun did not pass",
    )


def check_repository_link(href: str) -> None:
    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"}:
        return
    if href in {REPOSITORY, f"{REPOSITORY}/", f"{REPOSITORY}/issues"}:
        return
    for marker in ("/blob/main/", "/tree/main/"):
        prefix = f"{REPOSITORY}{marker}"
        if href.startswith(prefix):
            relative = unquote(href.removeprefix(prefix)).split("#", 1)[0]
            require(relative and (ROOT / relative).exists(), f"repository link target is missing: {href}")
            return
    require(
        parsed.netloc in {"tengzhang48.github.io", "doi.org"},
        f"unreviewed external site link: {href}",
    )


def main() -> None:
    require(INDEX.is_file() and STYLES.is_file() and EVIDENCE.is_file(), "site files are incomplete")
    actual_site_files = {path.name for path in SITE_ROOT.iterdir() if path.is_file()}
    require(actual_site_files == EXPECTED_SITE_FILES, f"unexpected site inventory: {sorted(actual_site_files)}")

    html = INDEX.read_text(encoding="utf-8")
    css = STYLES.read_text(encoding="utf-8")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    check_evidence(evidence)

    parser = SiteParser()
    parser.feed(html)
    require(parser.html_lang == "en", "html language must be English")
    require("CoupFE" in "".join(parser.title_parts), "page title must name CoupFE")
    require("width=device-width" in parser.meta.get("viewport", ""), "viewport metadata is missing")
    require(len(parser.meta.get("description", "")) >= 60, "meta description is too short")
    require(parser.landmarks.count("header") == 1, "site needs one header landmark")
    require(parser.landmarks.count("main") == 1, "site needs one main landmark")
    require(parser.landmarks.count("footer") == 1, "site needs one footer landmark")
    require(parser.landmarks.count("nav") >= 2, "site needs primary and footer navigation")
    require(all(parser.nav_labels), "every navigation landmark needs an aria-label")
    require(len(parser.ids) == len(set(parser.ids)), "element IDs must be unique")
    require(parser.section_ids == EXPECTED_SECTIONS, f"unexpected section order: {parser.section_ids}")
    require(sum(level == 1 for level, _text in parser.headings) == 1, "site needs exactly one H1")
    require(parser.headings and parser.headings[0][0] == 1, "the first heading must be H1")
    for previous, current in zip(parser.headings, parser.headings[1:]):
        require(current[0] <= previous[0] + 1, f"heading level jumps from {previous} to {current}")
    require(parser.script_count == 0, "the static Core site must not require JavaScript")
    allowed_figures = {
        HERTZ_SITE_FIGURE,
        LINEAR_BAR_SITE_FIGURE,
        NEO_HOOKEAN_BLOCK_SITE_FIGURE,
        CONTACT_3D_BLOCKS_SITE_FIGURE,
    }
    require(
        parser.image_count == len(allowed_figures),
        "site figure count must match the declared presentation artifacts",
    )
    require(parser.svg_image_count == 0, "do not duplicate figures inline as svg")
    require(
        parser.image_sources[0] == HERTZ_SITE_FIGURE,
        "the Hertz hero must be the first figure",
    )
    require(
        set(parser.image_sources) == allowed_figures,
        "every figure must be a declared presentation artifact",
    )
    require(parser.image_alts and parser.image_alts[0].strip(), "hero figure needs alt text")
    require((SITE_ROOT / HERTZ_SITE_FIGURE).is_file(), "representative figure file is missing")

    require('class="skip-link" href="#main"' in html, "skip link is missing")
    for href in parser.hrefs:
        if href.startswith("#"):
            require(href[1:] in set(parser.ids), f"local anchor has no destination: {href}")
        elif href in {"styles.css", "evidence.json"}:
            require((SITE_ROOT / href).is_file(), f"site asset is missing: {href}")
        else:
            check_repository_link(href)

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    author = project["authors"][0]["name"]
    normalized_html = " ".join(html.split())
    for required in (
        "A compact finite-element core for custom operators",
        "residual",
        "tangent",
        "commit",
        "U = Pq + U0",
        "CoupFE executes its own element interface",
        "Primary CoupFE path",
        "Optional external export",
        "Abaqus owns that external procedure",
        "not independent physical validation",
        "No general mesh-software adapter",
        "worked examples, not launch checks",
        "no retained final-revision multi-rank qualification",
        "scaling record",
        "does not run CoupFE in the browser",
        "CITATION.cff",
        "CREDITS.md",
        str(project["version"]),
        author,
        "1.533",
        "1.500",
        "1.01–1.06",
        "+1.3% to +6.2%",
        "0.857332",
        "1.04e-15",
        "2.02",
        "2.175e-02",
        "0.65",
        "2.503e-02",
        "Representative checked run",
        "Public record",
        "What is established here",
    ):
        require(required.casefold() in normalized_html.casefold(), f"required public text is missing: {required}")

    for forbidden in (
        "production-ready",
        "validated finite-element framework",
        "general MPI qualification",
        "two parallel backends",
        "build-time UEL, UMAT, and native-kernel generation",
        "UMAT",
        "scales to",
        "speedup",
    ):
        require(forbidden.casefold() not in html.casefold(), f"forbidden overclaim is present: {forbidden}")

    require(":focus-visible" in css, "visible keyboard focus styling is required")
    require("@media" in css, "responsive CSS is required")
    require("@import" not in css and "url(http" not in css.casefold(), "site must not load external CSS assets")
    print("site check passed: compact Core page, retained reruns, links, metadata, and claim boundaries")


if __name__ == "__main__":
    main()
