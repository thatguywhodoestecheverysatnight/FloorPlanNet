"""Vector geometry primitives and exporters (JSON, GeoJSON, SVG, DXF)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class Room:
    id: int
    label: str
    polygon: List[Point]
    holes: List[List[Point]] = field(default_factory=list)
    area_px: float = 0.0
    area_m2: Optional[float] = None
    centroid: Point = (0.0, 0.0)


@dataclass
class Wall:
    id: int
    p1: Point
    p2: Point
    thickness: float
    exterior: bool = False

    @property
    def length(self) -> float:
        return ((self.p1[0] - self.p2[0]) ** 2 + (self.p1[1] - self.p2[1]) ** 2) ** 0.5


@dataclass
class Opening:
    id: int
    kind: str                    # door | window
    polygon: List[Point]         # oriented rectangle, 4 points
    center: Point
    width: float                 # along the wall
    depth: float                 # across the wall
    angle: float                 # degrees, 0 = horizontal
    wall_id: Optional[int] = None


@dataclass
class FloorPlanGeometry:
    width: int
    height: int
    rooms: List[Room] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    wall_polygons: List[List[Point]] = field(default_factory=list)
    px_per_meter: Optional[float] = None
    meta: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        for w, wd in zip(self.walls, d["walls"]):
            wd["length"] = round(w.length, 2)
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "FloorPlanGeometry":
        g = FloorPlanGeometry(width=int(d["width"]), height=int(d["height"]),
                              px_per_meter=d.get("px_per_meter"), meta=dict(d.get("meta", {})))
        g.rooms = [Room(**{k: v for k, v in r.items()}) for r in d.get("rooms", [])]
        g.walls = [Wall(**{k: v for k, v in w.items() if k != "length"}) for w in d.get("walls", [])]
        g.openings = [Opening(**o) for o in d.get("openings", [])]
        g.wall_polygons = [list(map(tuple, p)) for p in d.get("wall_polygons", [])]
        return g

    def to_geojson(self) -> Dict[str, object]:
        """GeoJSON in image pixel space (y down). Set ``px_per_meter`` for metric areas."""
        feats: List[Dict[str, object]] = []

        def ring(pts: Sequence[Point]) -> List[List[float]]:
            r = [[float(x), float(y)] for x, y in pts]
            return r + [r[0]] if r and r[0] != r[-1] else r

        for r in self.rooms:
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(r.polygon)] + [ring(h) for h in r.holes]},
                          "properties": {"layer": "room", "id": r.id, "label": r.label, "area_px": r.area_px, "area_m2": r.area_m2}})
        for w in self.walls:
            feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [list(w.p1), list(w.p2)]},
                          "properties": {"layer": "wall", "id": w.id, "thickness": w.thickness, "exterior": w.exterior}})
        for o in self.openings:
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(o.polygon)]},
                          "properties": {"layer": o.kind, "id": o.id, "width": o.width, "wall_id": o.wall_id}})
        return {"type": "FeatureCollection", "features": feats,
                "properties": {"width": self.width, "height": self.height, "crs": "image-pixels"}}

    def to_svg(self, stroke_scale: float = 1.0) -> str:
        def pts(p: Sequence[Point]) -> str:
            return " ".join(f"{x:.1f},{y:.1f}" for x, y in p)

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
               f'width="{self.width}" height="{self.height}">',
               '<rect width="100%" height="100%" fill="#ffffff"/>', '<g id="rooms" fill="#81c7eb" fill-opacity="0.45" stroke="#2b7bb9" stroke-width="1">']
        for r in self.rooms:
            out.append(f'<polygon data-id="{r.id}" data-label="{r.label}" points="{pts(r.polygon)}"/>')
        out.append('</g><g id="walls" stroke="#262626" stroke-linecap="square">')
        for w in self.walls:
            out.append(f'<line data-id="{w.id}" x1="{w.p1[0]:.1f}" y1="{w.p1[1]:.1f}" x2="{w.p2[0]:.1f}" y2="{w.p2[1]:.1f}" '
                       f'stroke-width="{max(1.0, w.thickness) * stroke_scale:.1f}"/>')
        out.append('</g><g id="openings" stroke-width="1">')
        for o in self.openings:
            color = "#ec7063" if o.kind == "door" else "#58d68d"
            out.append(f'<polygon data-id="{o.id}" data-kind="{o.kind}" points="{pts(o.polygon)}" fill="{color}" stroke="{color}"/>')
        out.append("</g></svg>")
        return "\n".join(out)

    def to_dxf(self, path: str) -> None:
        """Write an AutoCAD DXF (R2010) with WALLS / ROOMS / DOORS / WINDOWS layers. Y is flipped to CAD convention."""
        import ezdxf

        doc = ezdxf.new("R2010")
        for name, color in (("WALLS", 7), ("ROOMS", 5), ("DOORS", 1), ("WINDOWS", 3)):
            doc.layers.add(name, color=color)
        msp = doc.modelspace()
        s = 1.0 / self.px_per_meter if self.px_per_meter else 1.0

        def tf(p: Point) -> Tuple[float, float]:
            return (p[0] * s, (self.height - p[1]) * s)

        for w in self.walls:
            msp.add_lwpolyline([tf(w.p1), tf(w.p2)], dxfattribs={"layer": "WALLS", "const_width": w.thickness * s})
        for r in self.rooms:
            msp.add_lwpolyline([tf(p) for p in r.polygon], close=True, dxfattribs={"layer": "ROOMS"})
            msp.add_text(r.label, dxfattribs={"layer": "ROOMS", "height": 8 * s}).set_placement(tf(r.centroid))
        for o in self.openings:
            msp.add_lwpolyline([tf(p) for p in o.polygon], close=True,
                               dxfattribs={"layer": "DOORS" if o.kind == "door" else "WINDOWS"})
        doc.saveas(path)

    def summary(self) -> Dict[str, object]:
        return {
            "rooms": len(self.rooms),
            "walls": len(self.walls),
            "doors": sum(o.kind == "door" for o in self.openings),
            "windows": sum(o.kind == "window" for o in self.openings),
            "total_wall_length_px": round(sum(w.length for w in self.walls), 1),
            "total_room_area_px": round(sum(r.area_px for r in self.rooms), 1),
        }
