#!/usr/bin/env python3
"""obsidian_graph.py — Куб-напарник из Obsidian заметок.

Архитектура:
  ObsidianGraph   — парсер Vault, построение графа, метрики
  GraphCube       — Tkinter overlay с 3D-визуализацией графа

Интеграция:
  from obsidian_graph import GraphCube
  graph = GraphCube(root)  # root = tk.Tk() from CubeApp
  graph.show()
  graph.hide()
  graph.toggle()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import math
import os
import re
import random
import tkinter as tk

# ── Forward declaration for type hints ──────────────────────────────────

WIKILINK_RE: re.Pattern = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]')
FRONTMATTER_RE: re.Pattern = re.compile(
    r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL | re.MULTILINE,
)

# ── Note data ──────────────────────────────────────────────────────────


class NoteNode:
    """A single note in the Obsidian graph."""

    def __init__(self, title: str, path: str, tags: List[str],
                 links: List[str]) -> None:
        self.title: str = title
        self.path: str = path
        self.tags: List[str] = tags
        self.links: List[str] = links  # outgoing wikilinks (titles)
        self.backlinks: int = 0  # incoming link count
        self.connectivity: float = 0.0  # 0..1 normalized


# ═══════════════════════════════════════════════════════════════════════════
# ObsidianGraph — парсер и граф
# ═══════════════════════════════════════════════════════════════════════════


class ObsidianGraph:
    """Build a graph of notes from an Obsidian vault."""

    def __init__(self, vault_path: str) -> None:
        self.vault_path: str = vault_path
        self.nodes: Dict[str, NoteNode] = {}  # path -> NoteNode
        self.edges: List[Tuple[str, str]] = []  # (source_path, target_path)
        self._loaded: bool = False

    # ── Public API ────────────────────────────────────────────────────

    def load(self) -> None:
        """Scan vault, parse all notes, build graph, compute metrics."""
        if self._loaded:
            return
        self.nodes.clear()
        self.edges.clear()
        self._scan_vault()
        self._resolve_links()
        self._compute_connectivity()
        self._loaded = True

    def reload(self) -> None:
        """Force re-scan."""
        self._loaded = False
        self.load()

    def get_connectivity(self, note_path: str) -> float:
        """Return 0..1 connectivity for a note."""
        node = self.nodes.get(note_path)
        return node.connectivity if node else 0.0

    def get_all_tags(self) -> Set[str]:
        """Return set of all tags across all notes."""
        tags: Set[str] = set()
        for node in self.nodes.values():
            tags.update(node.tags)
        return tags

    def get_notes_by_tag(self, tag: str) -> List[NoteNode]:
        """Return notes with the given tag."""
        return [n for n in self.nodes.values() if tag in n.tags]

    def get_top_notes(self, limit: int = 100) -> List[NoteNode]:
        """Return most connected notes, sorted by connectivity."""
        sorted_nodes: List[NoteNode] = sorted(
            self.nodes.values(),
            key=lambda n: n.connectivity,
            reverse=True,
        )
        return sorted_nodes[:limit]

    def get_path_by_title(self, title: str) -> Optional[str]:
        """Find note path by its title (first match)."""
        for path, node in self.nodes.items():
            if node.title == title or os.path.splitext(os.path.basename(path))[0] == title:
                return path
        return None

    # ── Internal ─────────────────────────────────────────────────────

    def _scan_vault(self) -> None:
        """Recursively find all .md files in the vault."""
        if not os.path.isdir(self.vault_path):
            return

        for root_dir, dirs, files in os.walk(self.vault_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            if '.obsidian' in root_dir or '.git' in root_dir:
                continue
            for fname in files:
                if fname.endswith('.md'):
                    fpath: str = os.path.join(root_dir, fname)
                    try:
                        self._parse_note(fpath)
                    except Exception:
                        pass  # skip malformed files

    def _parse_note(self, fpath: str) -> None:
        """Parse a single .md file into a NoteNode."""
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            content: str = f.read()

        # Frontmatter
        tags: List[str] = []
        fm_match = FRONTMATTER_MATCH(content)
        if fm_match:
            try:
                import yaml as _yaml
                fm_data = _yaml.safe_load(fm_match)
                if fm_data and 'tags' in fm_data:
                    raw_tags = fm_data['tags']
                    if isinstance(raw_tags, list):
                        tags = [str(t) for t in raw_tags]
                    elif isinstance(raw_tags, str):
                        tags = [raw_tags]
            except Exception:
                pass

        # Title from filename
        title: str = os.path.splitext(os.path.basename(fpath))[0]

        # Wikilinks
        links: List[str] = WIKILINK_RE.findall(content)
        links = [l.strip() for l in links if l.strip()]

        # Remove frontmatter from link search area if needed
        # (wikilinks in frontmatter shouldn't count as connections)
        # Simple: already parsed

        self.nodes[fpath] = NoteNode(
            title=title, path=fpath,
            tags=tags, links=links,
        )

    def _resolve_links(self) -> None:
        """Resolve wikilinks to actual file paths and build edges."""
        # Build title -> path map
        title_map: Dict[str, str] = {}
        for fpath, node in self.nodes.items():
            title_map[node.title] = fpath
            # Also map without extension
            base: str = os.path.splitext(os.path.basename(fpath))[0]
            title_map[base] = fpath

        # Resolve each link
        link_count: Dict[str, int] = {}  # path -> backlink count
        for fpath, node in self.nodes.items():
            for link_title in node.links:
                # Try exact title match first
                target: Optional[str] = title_map.get(link_title)
                if target is None:
                    # Try case-insensitive match
                    for t, p in title_map.items():
                        if t.lower() == link_title.lower():
                            target = p
                            break
                if target and target != fpath:
                    self.edges.append((fpath, target))
                    link_count[target] = link_count.get(target, 0) + 1

        # Assign backlinks to nodes
        for fpath, count in link_count.items():
            node = self.nodes.get(fpath)
            if node:
                node.backlinks = count

    def _compute_connectivity(self) -> None:
        """Compute normalized connectivity (0..1) for each note."""
        if not self.nodes:
            return

        # Connectivity = total_links + backlinks, normalized
        max_conn: int = 1
        conn_values: List[int] = []
        for node in self.nodes.values():
            conn: int = len(node.links) + node.backlinks
            conn_values.append(conn)
            if conn > max_conn:
                max_conn = conn

        for node in self.nodes.values():
            conn = len(node.links) + node.backlinks
            node.connectivity = conn / max_conn if max_conn > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Palette generation
# ═══════════════════════════════════════════════════════════════════════════

TAG_COLORS: Dict[str, Tuple[int, int, int]] = {}
_PALETTE: List[Tuple[int, int, int]] = [
    (255, 100, 100), (100, 200, 255), (100, 255, 100),
    (255, 200, 50),  (200, 100, 255), (50, 200, 200),
    (255, 150, 200), (150, 255, 150), (200, 200, 100),
    (100, 100, 255), (255, 100, 200), (100, 255, 200),
]


def _tag_to_color(tag: str) -> Tuple[int, int, int]:
    """Assign a consistent color to a tag."""
    if tag not in TAG_COLORS:
        h: int = hash(tag)
        TAG_COLORS[tag] = _PALETTE[abs(h) % len(_PALETTE)]
    return TAG_COLORS[tag]


def _third_color() -> Tuple[int, int, int]:
    """Return a random colour for untagged notes."""
    return _PALETTE[random.randrange(len(_PALETTE))]


# ═══════════════════════════════════════════════════════════════════════════
# GraphCube — Tkinter overlay for graph visualization
# ═══════════════════════════════════════════════════════════════════════════


class GraphCube:
    """Full-screen transparent overlay showing the Obsidian graph.

    Displays up to 100 notes as 3D particles on a sphere,
    with edges between connected notes.
    """

    TRANSPARENT_COLOR: str = '#000001'

    def __init__(self, root: tk.Tk,
                 vault_path: str = '',
                 on_note_click: Optional[Any] = None) -> None:
        self.root_ref: tk.Tk = root
        self.vault_path: str = vault_path or os.path.expanduser(
            r'~\Documents\Obsidian Vault')
        self.on_note_click: Optional[Any] = on_note_click
        self.graph: ObsidianGraph = ObsidianGraph(self.vault_path)
        self._visible: bool = False
        self._anim_running: bool = False
        self._time: float = 0.0
        self._hovered_note: Optional[str] = None
        self._rotation_y: float = 0.0
        self._rotation_x: float = 0.2

        # Cache projected data
        self._projected: Dict[str, Dict[str, Any]] = {}
        self._edge_lines: List[Tuple[float, float, float, float]] = []

        # ── Create window ────────────────────────────────────────────
        self.win = tk.Toplevel(root)
        self.win.title('♢ Graph Cube')
        self.win.overrideredirect(True)
        self.win.configure(bg=self.TRANSPARENT_COLOR)
        self.win.attributes('-transparentcolor', self.TRANSPARENT_COLOR)
        self.win.attributes('-topmost', True)

        sw: int = self.win.winfo_screenwidth()
        sh: int = self.win.winfo_screenheight()
        self.win.geometry(f'{sw}x{sh}+0+0')

        self.canvas = tk.Canvas(
            self.win, bg=self.TRANSPARENT_COLOR, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ── Mouse ────────────────────────────────────────────────────
        self.canvas.bind('<Motion>', self._on_motion)
        self.canvas.bind('<Button-1>', self._on_click)
        self.win.bind('<Escape>', lambda e: self.hide())
        self.win.withdraw()

    # ── Visibility ────────────────────────────────────────────────────

    def show(self) -> None:
        """Show the graph overlay and start animation."""
        if self._visible:
            return
        if not self.graph._loaded:
            self.graph.load()
        self._visible = True
        self.win.deiconify()
        self.win.lift()
        self.win.lift()
        self._anim_running = True
        self._anim_loop()

    def hide(self) -> None:
        """Hide the overlay."""
        self._visible = False
        self._anim_running = False
        self.win.withdraw()

    def toggle(self) -> None:
        """Flip visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()

    # ── Rendering ─────────────────────────────────────────────────────

    def _compute_projection(self) -> None:
        """Compute 3D positions of all notes and project to 2D."""
        top: List[NoteNode] = self.graph.get_top_notes(120)
        w: int = max(100, self.canvas.winfo_width())
        h: int = max(100, self.canvas.winfo_height())
        cx: float = w / 2.0
        cy: float = h / 2.0
        scale: float = min(w, h) * 0.35

        self._projected.clear()
        self._edge_lines.clear()

        # Fibonacci sphere for uniform distribution
        n: int = len(top)
        golden_angle: float = math.pi * (3.0 - math.sqrt(5.0))
        projected_nodes: Dict[str, Dict[str, Any]] = {}

        for i, node in enumerate(top):
            # Position on sphere
            r: float = 0.3 + 0.7 * (1.0 - node.connectivity)
            y: float = 1.0 - (i / max(1, n - 1)) * 2.0  # -1..1
            radius_at_y: float = math.sqrt(1.0 - y * y)
            theta: float = golden_angle * i + self._rotation_y
            phi: float = math.asin(y) + self._rotation_x

            x3d: float = radius_at_y * math.cos(theta) * r
            y3d: float = math.sin(phi) * r
            z3d: float = radius_at_y * math.sin(theta) * r

            # Project to 2D (simple orthographic with slight perspective)
            perspective: float = 1.0 / (1.5 + z3d * 0.5)
            sx: float = cx + x3d * scale * perspective
            sy: float = cy + y3d * scale * perspective

            # z-depth for painter's algorithm
            depth: float = z3d

            # Colour from first tag or fallback
            tag: str = node.tags[0] if node.tags else ''
            color: Tuple[int, int, int] = (
                _tag_to_color(tag) if tag else (200, 200, 220)
            )

            # Size: 2 + 6 * connectivity
            size: int = max(2, min(10, 2 + int(node.connectivity * 6)))

            projected_nodes[node.path] = {
                'x': sx,
                'y': sy,
                'depth': depth,
                'color': f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
                'size': size,
                'title': node.title,
                'path': node.path,
                'tags': node.tags,
                'connectivity': node.connectivity,
            }

        self._projected = projected_nodes

        # Build edge lines
        node_by_title: Dict[str, str] = {}
        for path, node in self.graph.nodes.items():
            node_by_title[node.title] = path

        for src_path, tgt_path in self.graph.edges[:600]:  # max edges
            src = projected_nodes.get(src_path)
            tgt = projected_nodes.get(tgt_path)
            if src and tgt:
                self._edge_lines.append(
                    (src['x'], src['y'], tgt['x'], tgt['y'])
                )

    def _anim_loop(self) -> None:
        """Animation loop: rotate + re-draw."""
        if not self._anim_running:
            return

        self._rotation_y += 0.008
        self._time += 0.05
        self._compute_projection()

        self.canvas.delete('all')

        # Sort by depth (far → near, painter's algorithm)
        sorted_notes: List[Dict[str, Any]] = sorted(
            self._projected.values(),
            key=lambda n: n['depth'],
        )

        # Draw edges (behind nodes)
        for ex1, ey1, ex2, ey2 in self._edge_lines:
            self.canvas.create_line(
                ex1, ey1, ex2, ey2,
                fill='#333355', width=1,
            )

        # Draw nodes
        for note in sorted_notes:
            x: float = note['x']
            y: float = note['y']
            size: int = note['size']
            color: str = note['color']
            title: str = note['title']

            # Main circle
            self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=color, outline='#444', width=1,
            )

            # Glow for highly connected notes
            if note['connectivity'] > 0.7:
                glow_size: int = size + 3
                self.canvas.create_oval(
                    x - glow_size, y - glow_size,
                    x + glow_size, y + glow_size,
                    fill='', outline=color, width=1, stipple='gray25',
                )

            # Hovered note title
            if title == self._hovered_note:
                self.canvas.create_text(
                    x, y - size - 8, text=title,
                    fill='#ffffff', font=('Segoe UI', 9),
                    anchor='s',
                )

        # Legend
        if hasattr(self, '_legend_items'):
            for item in self._legend_items:
                self.canvas.delete(item)

        self.win.after(50, self._anim_loop)  # ~20 fps

    # ── Mouse ─────────────────────────────────────────────────────────

    def _on_motion(self, event: tk.Event) -> None:
        """Detect hovered note."""
        mx: int = event.x
        my: int = event.y
        self._hovered_note = None
        for path, note in self._projected.items():
            if abs(note['x'] - mx) < note['size'] + 3 and \
               abs(note['y'] - my) < note['size'] + 3:
                self._hovered_note = note['title']
                break

    def _on_click(self, event: tk.Event) -> None:
        """Open clicked note in Obsidian."""
        mx: int = event.x
        my: int = event.y
        for path, note in self._projected.items():
            if abs(note['x'] - mx) < note['size'] + 5 and \
               abs(note['y'] - my) < note['size'] + 5:
                # Open in Obsidian
                rel_path: str = os.path.relpath(note['path'],
                                                 self.vault_path)
                obsidian_uri: str = (
                    f'obsidian://open?vault=Hermes Cube&file='
                    f'{rel_path.replace(os.sep, "/").replace(".md", "")}'
                )
                import subprocess as _sp
                try:
                    _sp.Popen(['start', obsidian_uri], shell=True)
                except Exception:
                    pass
                if self.on_note_click:
                    self.on_note_click(note['path'])
                break


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════


def FRONTMATTER_MATCH(content: str) -> str:
    """Extract YAML frontmatter from markdown content.

    Returns empty string if no valid frontmatter found.
    """
    import yaml as _yaml

    m = FRONTMATTER_RE.match(content)
    if m:
        fm_text: str = m.group(1)
        # Validate it's parseable YAML
        try:
            _yaml.safe_load(fm_text)
            return fm_text
        except Exception:
            pass
    return ''
