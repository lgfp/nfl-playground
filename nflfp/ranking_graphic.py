"""Ranking graphic: the 32 team logos in rank order, each with its rank number ("TIE" under tied numbers).

    uv run python -m nflfp.ranking_graphic "DEN,TEN,TB,CIN=DET,NE,WAS,KC,BUF,DAL=NYJ=PHI,MIA,..." -o ranking.png
    uv run python -m nflfp.ranking_graphic --file order.txt --bg "#101820" --number-color gold --font Inter:bold

The order is best to worst. Teams are written as abbreviation, nickname or full name (any case: `den`, `Broncos`,
`Denver Broncos`); `=` joins teams that are tied, so `DAL=NYJ=PHI` is a three-way tie for 10th and the next team is 13th.
Team names, colors and logo URLs come from nflverse (`data.load_teams`); logos are downloaded once into `.cache/`.

From Python:

    from nflfp.ranking_graphic import Style, make_graphic
    make_graphic(["DEN", "TEN", ["CIN", "DET"], "NE"], Style(background="#101820", number_color="gold")).save("r.png")

Styling lives in `Style`: background color or image, number color / font / size, tie label color / font / size / text,
canvas size, rows per column, and where logos come from. Fonts are a path to a .ttf/.otf file or an installed font name
(`Inter:bold` style names work where fontconfig exists, otherwise a file name Pillow can find).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from . import data

LOGO_COLUMNS = {"espn": "team_logo_espn", "squared": "team_logo_squared"}
EXTRA_ALIASES = {"wsh": "WAS", "jac": "JAX", "la": "LAR", "arz": "ARI"}  # common alternative codes
DEFAULT_FONTS = ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf")  # tried in order, then Pillow's built-in font
LOGO_SUFFIXES = (".png", ".webp", ".jpg", ".jpeg")


# --------------------------------------------------------------------------------------------------------- ranking

@dataclass(frozen=True)
class Entry:
    rank: int    # competition ranking: a 3-way tie for 10th is 10, 10, 10 and the next team is 13th
    team: str    # nflverse team abbreviation
    tie: bool


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def team_aliases(teams: pd.DataFrame) -> dict[str, str]:
    """Normalised abbreviation / nickname / full name -> team abbreviation."""
    out = {}
    for abbr, row in teams.iterrows():
        for key in (abbr, row["team_nick"], row["team_name"]):
            out[_norm(key)] = abbr
    return {**{k: v for k, v in EXTRA_ALIASES.items() if v in teams.index}, **out}


def parse_order(text: str) -> list:
    """'DEN, TEN, CIN=DET' (commas or newlines, `#` comments) -> ['DEN', 'TEN', ['CIN', 'DET']]."""
    items = []
    for line in text.splitlines():
        for token in line.split("#")[0].split(","):
            parts = [p.strip() for p in token.split("=") if p.strip()]
            if parts:
                items.append(parts if len(parts) > 1 else parts[0])
    return items


def rank_entries(order: Iterable, teams: pd.DataFrame) -> list[Entry]:
    """Entries in rank order. `order` items are a team (str) or a group of tied teams (list of str)."""
    aliases = team_aliases(teams)
    entries, rank, seen = [], 1, {}
    for item in order:
        group = [item] if isinstance(item, str) else list(item)
        if not group:
            raise ValueError("empty tie group")
        codes = []
        for name in group:
            code = aliases.get(_norm(name))
            if code is None:
                raise ValueError(f"unknown team {name!r} (try an abbreviation like DEN, a nickname or a full name; "
                                 f"--list-teams shows them all)")
            if code in seen:
                raise ValueError(f"{code} appears twice (as {seen[code]!r} and {name!r})")
            seen[code] = name
            codes.append(code)
        entries += [Entry(rank, c, len(codes) > 1) for c in codes]
        rank += len(codes)
    return entries


# ------------------------------------------------------------------------------------------------------------ style

@dataclass
class Style:
    background: str = "#f2f2f2"          # any color Pillow parses (name, #rrggbb, rgb(...)) or "transparent" (PNG only)
    background_image: str | None = None  # optional picture, scaled and cropped to fill the canvas (over `background`)
    background_image_opacity: float = 1.0  # < 1 washes the picture out toward `background`, to keep the numbers readable
    number_color: str = "#e02a2a"
    number_font: str | None = None       # path or installed font name; None = bold sans
    number_scale: float = 1.0            # 1.0 = a number about 46% of a row's height
    tie_color: str = "#000000"
    tie_font: str | None = None          # None = same as number_font
    tie_scale: float = 1.0
    tie_text: str = "TIE"
    width: int = 2000
    height: int = 1900
    rows: int = 8                        # teams per column; columns = ceil(teams / rows), filled top to bottom
    logo_source: str = "espn"            # nflverse logo column: "espn" (500 px, transparent) or "squared"
    logo_dir: str | None = None          # folder with <ABBR>.png files that take priority over downloads
    logo_overrides: dict[str, str] = field(default_factory=dict)  # {"HOU": "my_logo.png" or "https://..."} wins over all
    logo_scale: float = 1.0


def _color(spec: str) -> tuple[int, int, int, int]:
    if spec.strip().lower() == "transparent":
        return (0, 0, 0, 0)
    try:
        return ImageColor.getcolor(spec, "RGBA")
    except ValueError:
        raise ValueError(f"unknown color {spec!r} (use a name like 'gold' or a hex code like '#e02a2a')") from None


def _fc_match(name: str) -> str | None:
    """Font file for a fontconfig pattern such as 'Inter:bold', only if the family really matches."""
    exe = shutil.which("fc-match")
    if not exe:
        return None
    out = subprocess.run([exe, "-f", "%{family}|%{file}", name], capture_output=True, text=True).stdout
    family, _, path = out.rpartition("|")
    if name.split(":")[0].strip().lower() in family.lower() and Path(path).is_file():
        return path
    return None


def load_font(spec: str | None, size: float) -> ImageFont.FreeTypeFont:
    px = max(1, round(size))
    if spec is None:
        for name in DEFAULT_FONTS:
            try:
                return ImageFont.truetype(name, px)
            except OSError:
                continue
        return ImageFont.load_default(px)
    if Path(spec).is_file():
        return ImageFont.truetype(spec, px)
    try:
        return ImageFont.truetype(spec, px)          # Pillow searches the system font folders by file name
    except OSError:
        pass
    path = _fc_match(spec)
    if path:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            pass
    raise ValueError(f"font {spec!r} not found: pass a path to a .ttf/.otf file, or an installed font name")


# ------------------------------------------------------------------------------------------------------------ logos

def _logo_from(src: str, name: str, refresh: bool) -> Image.Image:
    """Open a local file, or download a URL into the cache first."""
    if re.match(r"https?://", src):
        src = str(data._fetch(src, name, refresh))  # cached for good: a logo URL's content does not change
    return Image.open(src).convert("RGBA")


def load_logo(team: str, teams: pd.DataFrame, style: Style, refresh: bool = False) -> Image.Image:
    """Logo for `team`: style.logo_overrides, then a file in style.logo_dir, then nflverse's logo URL."""
    tag = hashlib.md5(str(sorted(style.logo_overrides.items())).encode()).hexdigest()[:8]
    try:
        if team in style.logo_overrides:
            return _logo_from(style.logo_overrides[team], f"logo_custom_{team}_{tag}.png", refresh)
        if style.logo_dir:
            for f in Path(style.logo_dir).iterdir():
                if f.stem.upper() == team and f.suffix.lower() in LOGO_SUFFIXES:
                    return Image.open(f).convert("RGBA")
        if style.logo_source not in LOGO_COLUMNS:
            raise ValueError(f"logo_source must be one of {sorted(LOGO_COLUMNS)}")
        url = teams.loc[team, LOGO_COLUMNS[style.logo_source]]
        if style.logo_source == "espn":
            url = url.replace("/500-dark/", "/500/")  # nflverse links CAR's dark-background variant; the others are light
        return _logo_from(url, f"logo_{style.logo_source}_{team}.png", refresh)
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"could not get the {team} logo ({type(e).__name__}: {e}). If this machine cannot reach the "
                           f"logo host, put <ABBR>.png files in a folder and pass logo_dir / --logo-dir") from e


# ---------------------------------------------------------------------------------------------------------- drawing

def _draw_centered(draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str, font, fill, ref: str) -> float:
    """Draw `text` centered on (cx, cy): across its own ink, vertically on `ref`'s height so that every number in a row
    sits on the same baseline. Returns the baseline y."""
    left, _, right, _ = draw.textbbox((0, 0), text, font=font, anchor="ls")
    top, bottom = draw.textbbox((0, 0), ref, font=font, anchor="ls")[1::2]
    base = cy - (top + bottom) / 2
    draw.text((cx - (left + right) / 2, base), text, font=font, fill=fill, anchor="ls")
    return base


def draw_ranking(entries: list[Entry], logos: dict[str, Image.Image], style: Style) -> Image.Image:
    """Render the graphic. `logos` maps team abbreviation to an RGBA image."""
    W, H = style.width, style.height
    canvas = Image.new("RGBA", (W, H), _color(style.background))
    if style.background_image:
        pic = ImageOps.fit(Image.open(style.background_image).convert("RGBA"), (W, H), Image.LANCZOS)
        pic.putalpha(pic.getchannel("A").point(lambda a: round(a * min(1.0, max(0.0, style.background_image_opacity)))))
        canvas.alpha_composite(pic)

    n_cols = max(1, math.ceil(len(entries) / style.rows))
    pad_x, pad_y = 0.03 * W, 0.03 * H
    cw, ch = (W - 2 * pad_x) / n_cols, (H - 2 * pad_y) / style.rows
    num_font = load_font(style.number_font, 0.46 * ch * style.number_scale)
    tie_font = load_font(style.tie_font or style.number_font, 0.17 * ch * style.tie_scale)
    num_fill, tie_fill = _color(style.number_color), _color(style.tie_color)
    box_w, box_h = 0.52 * cw * style.logo_scale, 0.88 * ch * style.logo_scale

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))       # text goes on its own layer so a transparent color stays transparent
    draw = ImageDraw.Draw(layer)
    for i, e in enumerate(entries):
        col, row = divmod(i, style.rows)
        x0, cy = pad_x + col * cw, pad_y + (row + 0.5) * ch

        logo = logos[e.team]
        k = min(box_w / logo.width, box_h / logo.height)
        logo = logo.resize((max(1, round(logo.width * k)), max(1, round(logo.height * k))), Image.LANCZOS)
        canvas.alpha_composite(logo, (round(x0 + 0.62 * cw - logo.width / 2), round(cy - logo.height / 2)))

        base = _draw_centered(draw, x0 + 0.11 * cw, cy, str(e.rank), num_font, num_fill, ref="0")
        if e.tie:
            cap = -draw.textbbox((0, 0), "H", font=tie_font, anchor="ls")[1]
            _draw_centered(draw, x0 + 0.11 * cw, base + 0.03 * ch + cap / 2, style.tie_text, tie_font, tie_fill, ref="H")
    canvas.alpha_composite(layer)
    return canvas


def make_graphic(order: Iterable, style: Style | None = None, refresh: bool = False) -> Image.Image:
    """Ranking graphic for `order` (see module docstring); downloads / reads the logos as `style` says."""
    style = style or Style()
    teams = data.load_teams(refresh)
    entries = rank_entries(order, teams)
    return draw_ranking(entries, {e.team: load_logo(e.team, teams, style, refresh) for e in entries}, style)


def save(img: Image.Image, path: str | Path) -> None:
    if Path(path).suffix.lower() in (".jpg", ".jpeg"):   # JPEG has no alpha
        flat = Image.new("RGB", img.size, "white")
        flat.paste(img, mask=img.getchannel("A"))
        img = flat
    img.save(path)


# ----------------------------------------------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m nflfp.ranking_graphic", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("order", nargs="?", help="best to worst, comma separated; join ties with =  (e.g. 'DEN,TEN,CIN=DET,NE')")
    ap.add_argument("--file", help="read the order from a file instead (commas or one entry per line, # comments)")
    ap.add_argument("-o", "--out", default="ranking.png", help="output image (.png keeps transparency; .jpg does not)")
    ap.add_argument("--list-teams", action="store_true", help="print the team abbreviations and nicknames, then exit")
    d = Style()
    ap.add_argument("--bg", default=d.background, help="background color (name, #hex or 'transparent')")
    ap.add_argument("--bg-image", help="background picture, scaled and cropped to fill the canvas")
    ap.add_argument("--bg-image-opacity", type=float, default=d.background_image_opacity,
                    help="0-1; below 1 the picture fades toward --bg (a soft backdrop)")
    ap.add_argument("--number-color", default=d.number_color)
    ap.add_argument("--font", help="number font: path to .ttf/.otf, or an installed font name (e.g. 'Inter:bold')")
    ap.add_argument("--number-scale", type=float, default=d.number_scale)
    ap.add_argument("--tie-color", default=d.tie_color)
    ap.add_argument("--tie-font", help="tie label font (default: same as --font)")
    ap.add_argument("--tie-scale", type=float, default=d.tie_scale)
    ap.add_argument("--tie-text", default=d.tie_text)
    ap.add_argument("--width", type=int, default=d.width)
    ap.add_argument("--height", type=int, default=d.height)
    ap.add_argument("--rows", type=int, default=d.rows, help="teams per column")
    ap.add_argument("--logo-source", choices=sorted(LOGO_COLUMNS), default=d.logo_source)
    ap.add_argument("--logo-dir", help="folder of <ABBR>.png logos that take priority over downloaded ones")
    ap.add_argument("--logo", action="append", default=[], metavar="TEAM=PATH_OR_URL", help="logo for one team (repeatable)")
    ap.add_argument("--logo-scale", type=float, default=d.logo_scale)
    ap.add_argument("--refresh", action="store_true", help="re-download the team table and logos")
    a = ap.parse_args()

    teams = data.load_teams(a.refresh)
    if a.list_teams:
        for abbr, row in teams.iterrows():
            print(f"{abbr:4} {row['team_name']}")
        return
    if a.file:
        text = Path(a.file).read_text()
    elif a.order:
        text = a.order
    else:
        ap.error("give the order as an argument or with --file")

    overrides = {}
    for spec in a.logo:
        team, sep, src = spec.partition("=")
        code = team_aliases(teams).get(_norm(team))
        if not sep or code is None:
            ap.error(f"--logo wants TEAM=PATH_OR_URL with a known team, got {spec!r}")
        overrides[code] = src

    style = Style(background=a.bg, background_image=a.bg_image, background_image_opacity=a.bg_image_opacity, number_color=a.number_color, number_font=a.font,
                  number_scale=a.number_scale, tie_color=a.tie_color, tie_font=a.tie_font, tie_scale=a.tie_scale,
                  tie_text=a.tie_text, width=a.width, height=a.height, rows=a.rows, logo_source=a.logo_source,
                  logo_dir=a.logo_dir, logo_overrides=overrides, logo_scale=a.logo_scale)
    try:
        entries = rank_entries(parse_order(text), teams)
        missing = sorted(set(teams.index) - {e.team for e in entries})
        if missing:
            print(f"note: {len(entries)} of {len(teams)} teams given; not shown: {', '.join(missing)}", file=sys.stderr)
        img = draw_ranking(entries, {e.team: load_logo(e.team, teams, style, a.refresh) for e in entries}, style)
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f"error: {e}")
    save(img, a.out)
    print(f"wrote {a.out} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
