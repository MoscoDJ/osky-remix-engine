import { useEffect, useRef, useState } from "react";
import { Track } from "../api";

interface Props {
  tracks: Track[];
  onChange: (t: Track, upd: Partial<Pick<Track, "gain_db" | "pan" | "mute" | "solo">>) => void;
  onToggleLock: (t: Track) => void;
}

// Los deslizadores se mueven localmente y confirman UNA revisión: al soltar el mouse, o 600 ms
// después de la última tecla. Mientras hay un cambio pendiente, las recargas no pisan el valor.
export default function Mixer({ tracks, onChange, onToggleLock }: Props) {
  return (
    <div className="mixer">
      {tracks.map((t) => <Strip key={t.id} t={t} onChange={onChange} onToggleLock={onToggleLock} />)}
    </div>
  );
}

function useCommitted(value: number, commit: (v: number) => void) {
  const [v, setV] = useState(value);
  const pending = useRef(false);
  const timer = useRef<number | undefined>();
  const latest = useRef(v);
  latest.current = v;
  useEffect(() => { if (!pending.current) setV(value); }, [value]);
  const flush = () => {
    window.clearTimeout(timer.current);
    if (pending.current && latest.current !== value) commit(latest.current);
    pending.current = false;
  };
  const change = (x: number) => { pending.current = true; setV(x); };
  const keyUp = () => { window.clearTimeout(timer.current); timer.current = window.setTimeout(flush, 600); };
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return { v, change, flush, keyUp };
}

function Strip({ t, onChange, onToggleLock }: { t: Track } & Omit<Props, "tracks">) {
  const gain = useCommitted(t.gain_db, (x) => onChange(t, { gain_db: x }));
  const pan = useCommitted(t.pan, (x) => onChange(t, { pan: x }));
  const dis = t.locked;
  return (
    <div className={`strip ${t.kind} ${dis ? "locked" : ""}`} data-strip={t.id}>
      <div className="strip-name" title={t.kind}>{t.name}</div>
      <div className="strip-kind">{t.kind}</div>
      <input className="fader" type="range" min={-48} max={12} step={0.5} value={gain.v} disabled={dis}
             aria-label={`ganancia ${t.id}`} onChange={(e) => gain.change(+e.target.value)}
             onPointerUp={gain.flush} onKeyUp={gain.keyUp} onBlur={gain.flush} />
      <div className="db">{gain.v > 0 ? "+" : ""}{gain.v.toFixed(1)} dB</div>
      <input type="range" min={-1} max={1} step={0.05} value={pan.v} disabled={dis} aria-label={`paneo ${t.id}`}
             onChange={(e) => pan.change(+e.target.value)} onPointerUp={pan.flush} onKeyUp={pan.keyUp} onBlur={pan.flush} />
      <div className="db">pan {pan.v.toFixed(2)}</div>
      {t.kind !== "master" && (
        <div className="btns">
          <button className={t.mute ? "on" : ""} disabled={dis} onClick={() => onChange(t, { mute: !t.mute })}>M</button>
          <button className={t.solo ? "on" : ""} disabled={dis} onClick={() => onChange(t, { solo: !t.solo })}>S</button>
          <button onClick={() => onToggleLock(t)} title="Bloquear/desbloquear">{t.locked ? "🔒" : "🔓"}</button>
        </div>
      )}
      <div className="chain">{t.plugins.map((p) => <span key={p.id}>{p.uri.replace(".vst3", "")}</span>)}</div>
    </div>
  );
}
