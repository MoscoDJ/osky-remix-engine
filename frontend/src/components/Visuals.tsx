import { useEffect, useMemo, useRef, useState } from "react";

// Visualización del master (trabajo «visualize»): espectrograma, LUFS-S con marcadores de sección y
// objetivo del perfil, y correlación estéreo. Los datos vienen submuestreados del servidor; aquí
// solo se dibujan en canvas. El cursor sigue la reproducción del reproductor.

export interface VisData {
  duration_s: number; target_lufs: number | null; revision: string; render_key: string;
  sections: { label: string; start_s: number; end_s: number }[];
  spectrogram: { rows: number; cols: number; data_b64: string; freqs_hz: number[]; db_max: number; db_range: number };
  lufs_s: { t: number[]; lufs: (number | null)[]; window_s: number };
  correlation: { t: number[]; correlation: (number | null)[]; window_s: number;
                 sections: { label: string; correlation: number | null }[] };
}

const W = 1000;

function useCanvas(draw: (g: CanvasRenderingContext2D, w: number, h: number) => void, h: number, deps: unknown[]) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const g = c?.getContext("2d");
    if (!c || !g) return;
    g.clearRect(0, 0, W, h);
    draw(g, W, h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

// Rampa secuencial de un solo tono (oscuro → claro sobre fondo oscuro): magnitud, no identidad.
function ramp(v: number): [number, number, number] {
  const t = v / 255;
  return [Math.round(20 + 180 * t ** 1.6), Math.round(24 + 196 * t ** 1.1), Math.round(40 + 215 * t ** 0.7)];
}

function Markers({ data, testid }: { data: VisData; testid: string }) {
  return (
    <div className="vis-markers" data-testid={testid}>
      {data.sections.map((s) => (
        <span key={s.label + s.start_s} className="vis-marker" style={{ left: `${(s.start_s / data.duration_s) * 100}%` }}>
          {s.label}
        </span>
      ))}
    </div>
  );
}

function Cursor({ t, dur }: { t: number; dur: number }) {
  return t > 0 ? <div className="vis-cursor" style={{ left: `${Math.min(1, t / dur) * 100}%` }} /> : null;
}

interface Series { t: number[]; v: (number | null)[] }

function LineChart({ data, series, lo, hi, unit, h, guides, testid, now }:
  { data: VisData; series: Series; lo: number; hi: number; unit: string; h: number;
    guides: { v: number; label: string; dashed?: boolean }[]; testid: string; now: number }) {
  const [hover, setHover] = useState<{ x: number; t: number; v: number | null } | null>(null);
  const dur = data.duration_s;
  const y = (v: number) => h - 6 - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (h - 12);
  const ref = useCanvas((g, w) => {
    g.strokeStyle = "#2a2f3a"; g.lineWidth = 1;
    for (const s of data.sections) {                       // límites de sección
      const x = (s.start_s / dur) * w;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    }
    for (const gd of guides) {
      g.strokeStyle = gd.dashed ? "#ffb86b" : "#3a4050"; g.setLineDash(gd.dashed ? [6, 4] : []);
      g.beginPath(); g.moveTo(0, y(gd.v)); g.lineTo(w, y(gd.v)); g.stroke();
    }
    g.setLineDash([]);
    g.strokeStyle = "#7aa2ff"; g.lineWidth = 2; g.beginPath();
    let pen = false;
    series.t.forEach((t, i) => {
      const v = series.v[i];
      if (v === null) { pen = false; return; }
      const x = (t / dur) * w;
      if (pen) g.lineTo(x, y(v)); else g.moveTo(x, y(v));
      pen = true;
    });
    g.stroke();
  }, h, [data, series, lo, hi, h]);
  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    const t = frac * dur;
    if (!series.t.length) return;
    let i = 0, best = Infinity;
    series.t.forEach((tt, k) => { const d = Math.abs(tt - t); if (d < best) { best = d; i = k; } });
    setHover({ x: frac * 100, t: series.t[i], v: series.v[i] });
  };
  return (
    <div className="vis-plot" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <canvas ref={ref} width={W} height={h} data-testid={testid} data-points={series.t.length} />
      {guides.map((gd) => (
        <span key={gd.label} className="vis-guide" style={{ top: Math.max(0, y(gd.v) - 14) }}>{gd.label}</span>
      ))}
      <Cursor t={now} dur={dur} />
      {hover && <>
        <div className="vis-cross" style={{ left: `${hover.x}%` }} />
        <div className="vis-tip" style={{ left: `min(${hover.x}%, calc(100% - 140px))` }}>
          {hover.t.toFixed(1)} s · {hover.v === null ? "silencio" : `${hover.v.toFixed(2)} ${unit}`}
        </div>
      </>}
    </div>
  );
}

export default function Visuals({ data, now }: { data: VisData; now: number }) {
  const spec = data.spectrogram;
  const bytes = useMemo(() => Uint8Array.from(atob(spec.data_b64), (c) => c.charCodeAt(0)), [spec.data_b64]);
  const SH = 160;
  const specRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = specRef.current, g = c?.getContext("2d");
    if (!c || !g) return;
    const img = g.createImageData(spec.cols, spec.rows);
    for (let r = 0; r < spec.rows; r++) {
      for (let col = 0; col < spec.cols; col++) {
        const [R, G, B] = ramp(bytes[r * spec.cols + col]);
        const o = ((spec.rows - 1 - r) * spec.cols + col) * 4;   // fila 0 (graves) abajo
        img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
      }
    }
    const tmp = document.createElement("canvas");
    tmp.width = spec.cols; tmp.height = spec.rows;
    tmp.getContext("2d")!.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.drawImage(tmp, 0, 0, W, SH);
  }, [bytes, spec]);
  const freqTicks = [100, 1000, 10000].filter((f) => f > spec.freqs_hz[0] && f < spec.freqs_hz[spec.freqs_hz.length - 1]);
  const fy = (f: number) => {
    const lo = Math.log(spec.freqs_hz[0]), hi = Math.log(spec.freqs_hz[spec.freqs_hz.length - 1]);
    return SH - ((Math.log(f) - lo) / (hi - lo)) * SH;
  };

  const lufs = useMemo(() => ({ t: data.lufs_s.t, v: data.lufs_s.lufs }), [data]);
  const corr = useMemo(() => ({ t: data.correlation.t, v: data.correlation.correlation }), [data]);
  const vals = data.lufs_s.lufs.filter((v): v is number => v !== null);
  const top = Math.ceil(Math.max(-6, ...vals, data.target_lufs ?? -60) + 2);
  const bottom = Math.floor(Math.max(-60, Math.min(...vals, (data.target_lufs ?? 0) - 6) - 4));
  const guides = data.target_lufs !== null ? [{ v: data.target_lufs, label: `objetivo ${data.target_lufs} LUFS`, dashed: true }] : [];

  return (
    <div className="visuals" data-testid="visuals" data-render={data.render_key}>
      <div className="vis-block">
        <div className="vis-title">Espectrograma del master <span className="hint">(0 a −{spec.db_range} dB respecto del máximo)</span></div>
        <div className="vis-plot">
          <canvas ref={specRef} width={W} height={SH} data-testid="vis-spectrogram" data-rows={spec.rows} data-cols={spec.cols} />
          {freqTicks.map((f) => <span key={f} className="vis-guide" style={{ top: Math.max(0, fy(f) - 14) }}>{f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`}</span>)}
          <Cursor t={now} dur={data.duration_s} />
        </div>
        <Markers data={data} testid="vis-markers-spec" />
      </div>
      <div className="vis-block">
        <div className="vis-title">Sonoridad a corto plazo (LUFS-S, ventana {data.lufs_s.window_s} s)
          <span className="hint"> · eje {bottom} a {top} LUFS</span></div>
        <LineChart data={data} series={lufs} lo={bottom} hi={top} unit="LUFS" h={140} guides={guides} testid="vis-lufs" now={now} />
        <Markers data={data} testid="vis-markers-lufs" />
      </div>
      <div className="vis-block">
        <div className="vis-title">Correlación estéreo (ventana {data.correlation.window_s} s) <span className="hint">· +1 mono · 0 · −1 fase invertida</span></div>
        <LineChart data={data} series={corr} lo={-1} hi={1} unit="" h={100} testid="vis-corr" now={now}
                   guides={[{ v: 0, label: "0" }, { v: 1, label: "+1" }, { v: -1, label: "−1" }]} />
        <Markers data={data} testid="vis-markers-corr" />
        <table className="vis-table" data-testid="vis-corr-sections">
          <thead><tr><th>Sección</th><th>Correlación</th></tr></thead>
          <tbody>{data.correlation.sections.map((s) => (
            <tr key={s.label}><td>{s.label}</td><td>{s.correlation === null ? "silencio" : s.correlation.toFixed(2)}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}
