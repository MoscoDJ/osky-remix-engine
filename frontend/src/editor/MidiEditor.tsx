import { useEffect, useMemo, useState } from "react";
import { MidiClip, MidiNote } from "../api";

const S16 = 240;
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const noteName = (p: number) => `${NAMES[p % 12]}${Math.floor(p / 12) - 1}`;

interface Props {
  clip: MidiClip;
  locked: boolean;
  onSave: (notes: MidiNote[]) => void;
}

// Piano roll: los cambios se acumulan localmente y "Guardar" crea una sola revisión.
export default function MidiEditor({ clip, locked, onSave }: Props) {
  const [notes, setNotes] = useState<MidiNote[]>(clip.notes);
  useEffect(() => setNotes(clip.notes), [clip]);
  const dirty = useMemo(() => JSON.stringify(notes) !== JSON.stringify(clip.notes), [notes, clip.notes]);
  const pitches = notes.length ? notes.map((n) => n.pitch) : [60];
  const lo = Math.max(0, Math.min(...pitches) - 5);
  const hi = Math.min(127, Math.max(...pitches) + 5);
  const steps = Math.ceil(clip.length_ticks / S16);
  const cw = Math.max(6, Math.min(22, 1400 / steps));
  const ch = 12;
  const rows = hi - lo + 1;

  const toggleAt = (step: number, pitch: number) => {
    if (locked) return;
    const tick = step * S16;
    const hit = notes.findIndex((n) => n.pitch === pitch && n.tick <= tick && tick < n.tick + n.duration);
    if (hit >= 0) setNotes(notes.filter((_, i) => i !== hit));
    else setNotes([...notes, { tick, duration: S16, pitch, velocity: 100 }].sort((a, b) => a.tick - b.tick || a.pitch - b.pitch));
  };
  const transpose = (d: number) => {
    if (notes.some((n) => n.pitch + d < 0 || n.pitch + d > 127)) return;
    setNotes(notes.map((n) => ({ ...n, pitch: n.pitch + d })));
  };

  return (
    <div className="midi-editor">
      <div className="toolbar">
        <b>{clip.id}</b> · {notes.length} notas {locked && <span className="warn">· pista bloqueada (solo lectura)</span>}
        <button disabled={locked} onClick={() => transpose(-12)}>−8va</button>
        <button disabled={locked} onClick={() => transpose(-1)}>−1</button>
        <button disabled={locked} onClick={() => transpose(1)}>+1</button>
        <button disabled={locked} onClick={() => transpose(12)}>+8va</button>
        <button disabled={!dirty || locked} className="primary" onClick={() => onSave(notes)}>Guardar</button>
        <button disabled={!dirty} onClick={() => setNotes(clip.notes)}>Descartar</button>
      </div>
      <div className="roll-scroll">
        <svg width={40 + steps * cw} height={rows * ch} className="roll">
          {Array.from({ length: rows }, (_, r) => {
            const p = hi - r;
            const black = [1, 3, 6, 8, 10].includes(p % 12);
            return (
              <g key={p}>
                <rect x={40} y={r * ch} width={steps * cw} height={ch} className={black ? "row-black" : "row-white"} />
                {p % 12 === 0 && <text x={2} y={r * ch + 10} className="key-label">{noteName(p)}</text>}
              </g>
            );
          })}
          {Array.from({ length: steps + 1 }, (_, s) => (
            <line key={s} x1={40 + s * cw} x2={40 + s * cw} y1={0} y2={rows * ch} className={s % 16 === 0 ? "grid4" : s % 4 === 0 ? "grid" : "grid-fine"} />
          ))}
          <rect x={40} y={0} width={steps * cw} height={rows * ch} fill="transparent" data-roll="grid"
                onClick={(e) => {
                  const r = (e.currentTarget as SVGRectElement).getBoundingClientRect();
                  toggleAt(Math.floor((e.clientX - r.left) / cw), hi - Math.floor((e.clientY - r.top) / ch));
                }} />
          {notes.map((n, i) => (
            <rect key={i} x={40 + (n.tick / S16) * cw} y={(hi - n.pitch) * ch + 1} width={Math.max(2, (n.duration / S16) * cw - 1)}
                  height={ch - 2} className="note" data-note={`${n.tick}-${n.pitch}`}
                  onClick={() => !locked && setNotes(notes.filter((_, k) => k !== i))}>
              <title>{noteName(n.pitch)} · tick {n.tick} · vel {n.velocity}</title>
            </rect>
          ))}
        </svg>
      </div>
    </div>
  );
}
