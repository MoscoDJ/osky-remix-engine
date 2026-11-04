import { useEffect, useRef, useState } from "react";
import { api, AudioClip, Clip, MidiClip, State, Track } from "../api";

const ROW = 44;
const SECTION_COLORS = ["#3b4a6b", "#5a3b6b", "#6b3b45", "#3b6b5a", "#6b5a3b"];

interface Props {
  project: string;
  state: State;
  selected: string | null;
  onSelect: (clipId: string | null) => void;
  onMove: (clip: Clip, newTick: number) => void;
  onToggleLock: (t: Track) => void;
}

export default function Timeline({ project, state, selected, onSelect, onMove, onToggleLock }: Props) {
  const [pxPerBar, setPx] = useState(28);
  const [peaks, setPeaks] = useState<Record<string, { peaks: number[]; hop: number }>>({});
  const [drag, setDrag] = useState<{ id: string; x0: number; tick0: number; tick: number } | null>(null);
  const prj = state.project;
  const bar = state.bar_ticks;
  const tracks = prj.tracks.filter((t) => t.kind !== "master");
  const width = Math.max(1, Math.ceil(state.total_ticks / bar) + 2) * pxPerBar;
  const x = (tick: number) => (tick / bar) * pxPerBar;
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const shas = new Set<string>();
    tracks.forEach((t) => t.clips.forEach((c) => c.type === "audio" && shas.add(c.asset)));
    shas.forEach((s) => {
      if (!peaks[s]) api.peaks(project, s).then((d) => setPeaks((p) => ({ ...p, [s]: d })));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.revision, project]);

  const clipLen = (c: Clip) => (c.type === "midi" ? c.length_ticks : state.audio_clip_ticks[c.id] ?? bar);

  const startDrag = (e: React.PointerEvent, c: Clip, t: Track) => {
    onSelect(c.id);
    if (t.locked) return;
    (e.target as Element).setPointerCapture(e.pointerId);
    setDrag({ id: c.id, x0: e.clientX, tick0: c.start_tick, tick: c.start_tick });
  };
  const moveDrag = (e: React.PointerEvent) => {
    if (!drag) return;
    const snap = e.shiftKey ? bar : prj.ppq;
    const raw = drag.tick0 + ((e.clientX - drag.x0) / pxPerBar) * bar;
    setDrag({ ...drag, tick: Math.max(0, Math.round(raw / snap) * snap) });
  };
  const endDrag = () => {
    if (!drag) return;
    const c = tracks.flatMap((t) => t.clips).find((k) => k.id === drag.id);
    if (c && drag.tick !== drag.tick0) onMove(c, drag.tick);
    setDrag(null);
  };

  const markers = [...prj.markers].sort((a, b) => a.tick - b.tick);
  return (
    <div className="timeline">
      <div className="toolbar">
        <span>Zoom</span>
        <input type="range" min={8} max={80} value={pxPerBar} onChange={(e) => setPx(+e.target.value)} aria-label="zoom" />
        <span className="hint">Arrastra un clip para moverlo (ajuste al tiempo; Shift = al compás). Clic = seleccionar.</span>
      </div>
      <div className="tl-scroll">
        <div className="tl-heads">
          <div className="tl-ruler-head">Secciones</div>
          {tracks.map((t) => (
            <div key={t.id} className={`tl-head ${t.locked ? "locked" : ""}`} style={{ height: ROW }} data-track={t.id}>
              <span className="name">{t.name}</span>
              <button className="lock" title={t.locked ? "Desbloquear" : "Bloquear (protege tus ediciones)"}
                      onClick={() => onToggleLock(t)} aria-label={`bloqueo ${t.id}`}>{t.locked ? "🔒" : "🔓"}</button>
            </div>
          ))}
        </div>
        <svg ref={svgRef} width={width} height={28 + tracks.length * ROW} className="tl-svg"
             onPointerMove={moveDrag} onPointerUp={endDrag} onClick={(e) => e.target === svgRef.current && onSelect(null)}>
          {markers.map((m, i) => {
            const end = i + 1 < markers.length ? markers[i + 1].tick : state.total_ticks;
            return (
              <g key={m.id}>
                <rect x={x(m.tick)} y={0} width={Math.max(1, x(end) - x(m.tick))} height={24}
                      fill={SECTION_COLORS[i % SECTION_COLORS.length]} />
                <text x={x(m.tick) + 4} y={16} className="sec-label">{m.label}</text>
              </g>
            );
          })}
          {Array.from({ length: Math.ceil(width / pxPerBar) }, (_, b) => (
            <line key={b} x1={b * pxPerBar} x2={b * pxPerBar} y1={24} y2={28 + tracks.length * ROW}
                  className={b % 4 === 0 ? "grid4" : "grid"} />
          ))}
          {tracks.map((t, row) => (
            <g key={t.id} transform={`translate(0, ${28 + row * ROW})`}>
              <rect x={0} y={0} width={width} height={ROW - 2} className="lane" />
              {t.clips.map((c) => {
                const tick = drag && drag.id === c.id ? drag.tick : c.start_tick;
                const w = Math.max(3, x(clipLen(c)));
                return (
                  <g key={c.id} transform={`translate(${x(tick)}, 3)`} onPointerDown={(e) => startDrag(e, c, t)}
                     className={`clip ${selected === c.id ? "sel" : ""} ${t.locked ? "locked" : ""}`} data-clip={c.id}>
                    <rect width={w} height={ROW - 8} rx={4} className={c.type === "audio" ? "clip-audio" : "clip-midi"} />
                    {c.type === "midi" ? <MiniNotes clip={c} w={w} h={ROW - 8} /> :
                      <Wave clip={c} w={w} h={ROW - 8} pk={peaks[c.asset]} />}
                    <text x={4} y={12} className="clip-label">{c.id}</text>
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}

function MiniNotes({ clip, w, h }: { clip: MidiClip; w: number; h: number }) {
  if (!clip.notes.length) return null;
  const lo = Math.min(...clip.notes.map((n) => n.pitch));
  const hi = Math.max(...clip.notes.map((n) => n.pitch));
  const span = Math.max(1, hi - lo + 1);
  return (
    <g>
      {clip.notes.slice(0, 600).map((n, i) => (
        <rect key={i} x={(n.tick / clip.length_ticks) * w} y={16 + (1 - (n.pitch - lo + 1) / span) * (h - 20)}
              width={Math.max(1, (n.duration / clip.length_ticks) * w)} height={Math.max(1.5, (h - 20) / span)} className="mini-note" />
      ))}
    </g>
  );
}

function Wave({ clip, w, h, pk }: { clip: AudioClip; w: number; h: number; pk?: { peaks: number[]; hop: number } }) {
  if (!pk) return null;
  const a = Math.floor(clip.source_offset / pk.hop);
  const b = Math.ceil((clip.source_offset + clip.source_length) / pk.hop);
  const seg = pk.peaks.slice(a, b);
  const n = Math.min(seg.length, Math.max(2, Math.floor(w)));
  const mid = 8 + (h - 8) / 2;
  const pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const v = seg[Math.floor((i / n) * seg.length)] ?? 0;
    const xx = (i / n) * w;
    pts.push(`M${xx},${mid - v * (h - 10) / 2}L${xx},${mid + v * (h - 10) / 2}`);
  }
  return <path d={pts.join("")} className="wave" />;
}
