import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import { Project, secondsAt } from "../api";

const COLORS = ["rgba(90,120,200,.18)", "rgba(160,90,200,.18)", "rgba(200,90,110,.18)", "rgba(90,200,150,.18)"];

export function useWave(url: string | null, prj: Project | null, height = 90) {
  const ref = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const [ready, setReady] = useState(false);
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!ref.current || !url) return;
    const regions = RegionsPlugin.create();
    const w = WaveSurfer.create({ container: ref.current, height, waveColor: "#7aa2ff", progressColor: "#c8d6ff",
                                  cursorColor: "#fff", normalize: true, plugins: [regions] });
    ws.current = w;
    setReady(false);
    w.load(url);
    w.on("ready", () => {
      setReady(true);
      if (prj) {
        const ms = [...prj.markers].sort((a, b) => a.tick - b.tick);
        ms.forEach((m, i) => {
          const end = i + 1 < ms.length ? secondsAt(ms[i + 1].tick, prj) : w.getDuration();
          regions.addRegion({ start: secondsAt(m.tick, prj), end, content: m.label, color: COLORS[i % COLORS.length],
                              drag: false, resize: false });
        });
      }
    });
    w.on("timeupdate", (s) => setT(s));
    return () => { w.destroy(); ws.current = null; };
  }, [url, prj, height]);
  return { ref, ws, ready, t };
}

export default function Player({ url, prj, info }: { url: string | null; prj: Project | null; info?: string }) {
  const { ref, ws, ready, t } = useWave(url, prj);
  return (
    <div className="player">
      <div className="toolbar">
        <button disabled={!ready} onClick={() => ws.current?.playPause()} data-testid="play">▶︎ / ❚❚</button>
        <span className="time">{t.toFixed(1)} s</span>
        {info && <span className="hint">{info}</span>}
      </div>
      {url ? <div ref={ref} className="wave-box" data-testid="master-wave" data-ready={ready ? "1" : "0"} />
           : <div className="empty">Renderiza para escuchar la revisión actual.</div>}
    </div>
  );
}
