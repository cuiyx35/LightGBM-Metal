#!/usr/bin/env python3
"""Render this repository's GitHub star history as light/dark SVGs.

The workflow uses its short-lived GITHUB_TOKEN. Only aggregated star dates are
written to SVGs; the token and stargazer identities are never written to disk.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from xml.sax.saxutils import escape

GRAPHQL_URL = "https://api.github.com/graphql"
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    createdAt
    stargazerCount
    stargazers(
      first: 100
      after: $cursor
      orderBy: {field: STARRED_AT, direction: ASC}
    ) {
      edges { starredAt }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


def parse_date(value: str) -> dt.date:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def fetch_history(repository: str, token: str) -> tuple[dt.date, list[dt.date]]:
    owner, name = repository.split("/", 1)
    cursor = None
    created = None
    expected = None
    dates: list[dt.date] = []

    while True:
        body = json.dumps(
            {
                "query": QUERY,
                "variables": {"owner": owner, "name": name, "cursor": cursor},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            GRAPHQL_URL,
            data=body,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "LightGBM-Metal-star-history",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"GitHub GraphQL returned HTTP {exc.code}") from exc

        if payload.get("errors"):
            messages = "; ".join(error.get("message", "unknown error") for error in payload["errors"])
            raise RuntimeError(f"GitHub GraphQL error: {messages}")
        repo = payload.get("data", {}).get("repository")
        if not repo:
            raise RuntimeError(f"Repository {repository} is not available to this token")

        created = parse_date(repo["createdAt"])
        expected = repo["stargazerCount"]
        connection = repo["stargazers"]
        dates.extend(parse_date(edge["starredAt"]) for edge in connection["edges"])
        page_info = connection["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        next_cursor = page_info["endCursor"]
        if not next_cursor or next_cursor == cursor:
            raise RuntimeError("GitHub GraphQL pagination did not advance")
        cursor = next_cursor

    if len(dates) != expected:
        raise RuntimeError(
            f"GitHub returned {len(dates)} star dates but reports {expected} stars"
        )
    return created, sorted(dates)


def render_svg(
    repository: str,
    created: dt.date,
    starred: list[dt.date],
    as_of: dt.date,
    dark: bool,
) -> str:
    width, height = 860, 360
    left, right, top, bottom = 76, 32, 78, 62
    plot_width, plot_height = width - left - right, height - top - bottom
    end = max(as_of, created)
    span = max(1, (end - created).days)
    plot_end = created + dt.timedelta(days=span)
    count = len(starred)
    ceiling = max(1, count)

    colors = (
        {
            "background": "#0d1117",
            "text": "#e6edf3",
            "muted": "#8b949e",
            "grid": "#30363d",
            "line": "#58a6ff",
            "fill": "#58a6ff",
        }
        if dark
        else {
            "background": "#ffffff",
            "text": "#24292f",
            "muted": "#57606a",
            "grid": "#d8dee4",
            "line": "#0969da",
            "fill": "#0969da",
        }
    )

    def x_position(day: dt.date) -> float:
        offset = min(max((day - created).days, 0), span)
        return left + plot_width * offset / span

    def y_position(stars: int) -> float:
        return top + plot_height * (1 - stars / ceiling)

    per_day = collections.Counter(starred)
    points = [(x_position(created), y_position(0))]
    cumulative = 0
    for day in sorted(per_day):
        x = x_position(day)
        points.append((x, y_position(cumulative)))
        cumulative += per_day[day]
        points.append((x, y_position(cumulative)))
    points.append((x_position(plot_end), y_position(cumulative)))
    line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    ticks = sorted({0, math.ceil(count / 2), ceiling})
    grid_lines = []
    for tick in ticks:
        y = y_position(tick)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            f'stroke="{colors["grid"]}" stroke-width="1"/>'
        )
        grid_lines.append(
            f'<text x="{left-13}" y="{y+4:.1f}" text-anchor="end" '
            f'fill="{colors["muted"]}" font-size="13">{tick}</text>'
        )

    x_labels = []
    for day, anchor, x in [
        (created, "start", left),
        (end, "end", width - right),
    ]:
        label = "launch day" if anchor == "end" and end == created else day.isoformat()
        x_labels.append(
            f'<text x="{x:.1f}" y="{height-28}" text-anchor="{anchor}" '
            f'fill="{colors["muted"]}" font-size="13">{label}</text>'
        )

    safe_name = escape(repository)
    title = f"{safe_name} · {count} stars"
    description = (
        f"Daily GitHub star history through {end.isoformat()}, "
        f"generated for this repository by its own GitHub Actions workflow."
    )
    empty_note = (
        f'<text x="{width/2:.1f}" y="{top+plot_height/2:.1f}" text-anchor="middle" '
        f'fill="{colors["muted"]}" font-size="15">The curve starts with the first star</text>'
        if count == 0
        else ""
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
            f"<title>{title}</title>",
            f"<desc>{escape(description)}</desc>",
            f'<rect width="{width}" height="{height}" rx="12" fill="{colors["background"]}"/>',
            f'<text x="{left}" y="37" fill="{colors["text"]}" font-size="20" '
            f'font-family="system-ui, sans-serif" font-weight="600">Star History</text>',
            f'<text x="{left}" y="59" fill="{colors["muted"]}" font-size="13" '
            f'font-family="system-ui, sans-serif">{safe_name} · {count} stars</text>',
            '<g font-family="system-ui, sans-serif">',
            *grid_lines,
            *x_labels,
            f'<polyline points="{line_points}" fill="none" stroke="{colors["line"]}" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
            f'<circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="4" '
            f'fill="{colors["fill"]}"/>',
            empty_note,
            "</g>",
            "</svg>",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--output-dir", default="assets")
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Local JSON with created_at and starred_at dates; no API call",
    )
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=dt.datetime.now(dt.timezone.utc).date(),
        help="Chart end date for a reproducible local render (YYYY-MM-DD)",
    )
    args = parser.parse_args()
    if not args.repo or not REPOSITORY_PATTERN.fullmatch(args.repo):
        parser.error("--repo must be an owner/name repository slug")

    if args.fixture:
        fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        created = parse_date(fixture["created_at"])
        starred = sorted(parse_date(value) for value in fixture["starred_at"])
    else:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            parser.error("GITHUB_TOKEN is required unless --fixture is provided")
        created, starred = fetch_history(args.repo, token)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dark, filename in (
        (False, "star-history.svg"),
        (True, "star-history-dark.svg"),
    ):
        (output_dir / filename).write_text(
            render_svg(args.repo, created, starred, args.as_of, dark),
            encoding="utf-8",
        )
    print(f"Rendered {len(starred)} stars through {args.as_of.isoformat()}")


if __name__ == "__main__":
    main()
