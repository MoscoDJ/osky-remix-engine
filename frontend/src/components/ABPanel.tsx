import { useEffect, useState } from "react";
import { Project, runJob } from "../api";
import { useWave } from "./Player";

interface Props { project: string; branches: string[]; prj: Project | null; profile: string }

// A/B igualado en sonoridad: dos formas de onda sincronizadas; el botón cambia cuál suena.
export default function ABPanel({ project, branches, prj, profile }: Props) {
  const [a, setA] = useState(branches[0] ?? "main");
  const [b, setB] = useState(branches[1] ?? branches[0] ?? "main");
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [listen, setListen] = useState<"A" | "B">("A");
  const A = useWave(res?.a_url ?? null, prj, 60);
  const B = useWave(res?.b_url ?? null, prj, 60);

  useEffect(() => {
    if (!A.ws.current || !B.ws.current) return;
    A.ws.current.setVolume(listen === "A" ? 1 : 0);
    B.ws.current.setVolume(listen === "B" ? 1 : 0);
  }, [listen, A.ready, B.ready, A.ws, B.ws]);

  const run = async () => {
    setBusy(true); setErr(null);
    try { setRes(await runJob(project, { kind: "compare", a, b, profile })); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  };
  const play = () => {
    const wa = A.ws.current, wb = B.ws.current;
    if (!wa || !wb) return;
    if (wa.isPlaying()) { wa.pause(); wb.pause(); return; }
    wb.setTime(wa.getCurrentTime());
    wa.play(); wb.play();
  };
  const toggle = () => {
    const wa = A.ws.current, wb = B.ws.current;
    if (wa && wb) wb.setTime(wa.getCurrentTime());
    setListen(listen === "A" ? "B" : "A");
  };

  return (
    <div className="ab">
      <div className="toolbar">
        A <select value={a} onChange={(e) => setA(e.target.value)} aria-label="A">{branches.map((x) => <option key={x} value={x}>{x}</option>)}</select>
        B <select value={b} onChange={(e) => setB(e.target.value)} aria-label="B">{branches.map((x) => <option key={x} value={x}>{x}</option>)}</select>
        <button className="primary" disabled={busy} onClick={run}>{busy ? "Comparando…" : "Comparar"}</button>
        {res && <>
          <button onClick={play} disabled={!A.ready || !B.ready}>▶︎ / ❚❚</button>
          <button onClick={toggle} className="ab-toggle" data-testid="ab-toggle">Escuchando {listen} (cambiar)</button>
        </>}
        {err && <span className="warn">{err}</span>}
      </div>
      {res && (
        <div className="ab-body" data-testid="ab-report">
          <div className={`ab-wave ${listen === "A" ? "on" : ""}`}><span>A · {a}</span><div ref={A.ref} /></div>
          <div className={`ab-wave ${listen === "B" ? "on" : ""}`}><span>B · {b} (igualado {res.report.b_gain_to_match_db > 0 ? "+" : ""}{res.report.b_gain_to_match_db} dB)</span><div ref={B.ref} /></div>
          <div className="ab-grid">
            <table>
              <thead><tr><th>Banda</th><th>B − A</th></tr></thead>
              <tbody>{Object.entries(res.report.bands_db_b_minus_a).map(([k, v]: any) => <tr key={k}><td>{k}</td><td>{v > 0 ? "+" : ""}{v} dB</td></tr>)}</tbody>
            </table>
            {res.report.sections && (
              <table>
                <thead><tr><th>Sección</th><th>A LUFS</th><th>B LUFS</th></tr></thead>
                <tbody>{res.report.sections.map((s: any) => <tr key={s.section}><td>{s.section}</td><td>{s.lufs_a}</td><td>{s.lufs_b_matched}</td></tr>)}</tbody>
              </table>
            )}
            <div className="diff">
              <b>Cambios ({res.diff.length})</b>
              <ul>{res.diff.slice(0, 60).map((d: string, i: number) => <li key={i}>{d}</li>)}</ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
