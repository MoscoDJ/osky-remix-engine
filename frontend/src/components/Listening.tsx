import { useCallback, useEffect, useState } from "react";
import { api, Revision, runJob, Verdict, ListeningSummary } from "../api";
import { useWave } from "./Player";

interface Props { project: string; branches: string[]; log: Revision[]; profile: string }

// Escucha ciega: el servidor sortea qué versión es X y cuál Y (igualadas en sonoridad). El usuario
// elige preferencia y califica; al guardar se revela la identidad y queda en el registro.
export default function Listening({ project, branches, log, profile }: Props) {
  const refs = [...branches.map((b) => ({ v: b, label: `rama ${b}` })),
                ...log.map((r) => ({ v: r.id, label: `${r.id.slice(0, 8)} · ${r.message.slice(0, 40)}` }))];
  const [a, setA] = useState(branches[0] ?? "main");
  const [b, setB] = useState(branches[1] ?? branches[0] ?? "main");
  const [sess, setSess] = useState<{ session: string; x_url: string; y_url: string; profile: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [listen, setListen] = useState<"X" | "Y">("X");
  const [pref, setPref] = useState<"x" | "y" | "none" | null>(null);
  const [rx, setRx] = useState(3);
  const [ry, setRy] = useState(3);
  const [note, setNote] = useState("");
  const [reveal, setReveal] = useState<Verdict | null>(null);
  const [hist, setHist] = useState<{ verdicts: Verdict[]; summary: ListeningSummary } | null>(null);
  const X = useWave(sess?.x_url ?? null, null, 50);
  const Y = useWave(sess?.y_url ?? null, null, 50);

  const loadHist = useCallback(() => { api.listening(project).then(setHist).catch((e) => setErr(e.message)); }, [project]);
  useEffect(() => { loadHist(); }, [loadHist]);

  useEffect(() => {
    if (!X.ws.current || !Y.ws.current) return;
    X.ws.current.setVolume(listen === "X" ? 1 : 0);
    Y.ws.current.setVolume(listen === "Y" ? 1 : 0);
  }, [listen, X.ready, Y.ready, X.ws, Y.ws]);

  const prepare = async () => {
    setBusy(true); setErr(null); setReveal(null); setPref(null); setRx(3); setRy(3); setNote(""); setListen("X");
    try { setSess(await runJob(project, { kind: "blind", a, b, profile })); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  };
  const play = () => {
    const wx = X.ws.current, wy = Y.ws.current;
    if (!wx || !wy) return;
    if (wx.isPlaying()) { wx.pause(); wy.pause(); return; }
    wy.setTime(wx.getCurrentTime());
    wx.play(); wy.play();
  };
  const toggle = () => {
    const wx = X.ws.current, wy = Y.ws.current;
    if (wx && wy) wy.setTime(wx.getCurrentTime());
    setListen(listen === "X" ? "Y" : "X");
  };
  const save = async () => {
    if (!sess || !pref) return;
    try {
      X.ws.current?.pause(); Y.ws.current?.pause();
      const v = await api.verdict(project, { session: sess.session, preferred: pref, rating_x: rx, rating_y: ry, note });
      setReveal(v); setSess(null); loadHist();
    } catch (e: any) { setErr(e.message); }
  };
  const label = (v: Verdict, w: "a" | "b") => `${w === "a" ? v.label_a : v.label_b} (${(w === "a" ? v.rev_a : v.rev_b).slice(0, 8)})`;
  const Stars = ({ value, set, name }: { value: number; set: (n: number) => void; name: string }) => (
    <span className="stars" role="radiogroup" aria-label={`calificación ${name}`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} className={n <= value ? "on" : ""} aria-label={`${name} ${n}`} aria-checked={n === value} role="radio"
                onClick={() => set(n)}>{n}</button>
      ))}
    </span>
  );
  const fmt = (s: number | null) => (s === null ? "—" : s.toFixed(2));
  const s = hist?.summary;

  return (
    <div className="listening">
      <div className="toolbar">
        A <select value={a} onChange={(e) => setA(e.target.value)} aria-label="escucha A">{refs.map((x) => <option key={"a" + x.v} value={x.v}>{x.label}</option>)}</select>
        B <select value={b} onChange={(e) => setB(e.target.value)} aria-label="escucha B">{refs.map((x) => <option key={"b" + x.v} value={x.v}>{x.label}</option>)}</select>
        <button className="primary" disabled={busy || a === b} onClick={prepare}>{busy ? "Preparando…" : "Preparar escucha ciega"}</button>
        <span className="hint">perfil {profile} · orden sorteado e igualado en sonoridad</span>
        {err && <span className="warn">{err}</span>}
      </div>
      {sess && (
        <div className="blind" data-testid="blind-session">
          <div className="toolbar">
            <button onClick={play} disabled={!X.ready || !Y.ready} data-testid="blind-play">▶︎ / ❚❚</button>
            <button onClick={toggle} className="ab-toggle" data-testid="blind-toggle">Escuchando {listen} (cambiar)</button>
          </div>
          <div className={`ab-wave ${listen === "X" ? "on" : ""}`}><span>X</span><div ref={X.ref} data-testid="wave-x" data-ready={X.ready ? "1" : "0"} /></div>
          <div className={`ab-wave ${listen === "Y" ? "on" : ""}`}><span>Y</span><div ref={Y.ref} data-testid="wave-y" data-ready={Y.ready ? "1" : "0"} /></div>
          <div className="verdict-form">
            <div>Prefiero:
              {(["x", "y", "none"] as const).map((p) => (
                <label key={p}><input type="radio" name="pref" checked={pref === p} onChange={() => setPref(p)} aria-label={`prefiero ${p}`} />
                  {p === "none" ? "sin preferencia" : p.toUpperCase()}</label>
              ))}
            </div>
            <div>X <Stars value={rx} set={setRx} name="X" /> · Y <Stars value={ry} set={setRy} name="Y" /></div>
            <textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="Nota (opcional)" aria-label="nota" rows={2} />
            <button className="primary" disabled={!pref} onClick={save}>Guardar y revelar</button>
          </div>
        </div>
      )}
      {reveal && (
        <div className="reveal" data-testid="reveal">
          X era <b>{reveal.x_was.toUpperCase()}</b> · {label(reveal, reveal.x_was)}; Y era <b>{reveal.x_was === "a" ? "B" : "A"}</b> · {label(reveal, reveal.x_was === "a" ? "b" : "a")}.{" "}
          {reveal.preferred === "none" ? "Sin preferencia." : <>Preferiste <b>{reveal.preferred.toUpperCase()}</b> · {label(reveal, reveal.preferred)}.</>}{" "}
          Evaluador: A {fmt(reveal.score_a)} · B {fmt(reveal.score_b)} (menor = mejor).
        </div>
      )}
      <h4>Registro de escucha</h4>
      {s && (
        <div className="hint" data-testid="listening-summary">
          veredictos: {s.verdicts} · acuerdo con el evaluador {s.agreement_pct === null ? "— (sin pares comparables)" : `${s.agreement_pct} % (${s.agree}/${s.pairs})`}
          {s.mean_rating_preferred !== null && ` · calificación media preferida ${s.mean_rating_preferred} / otra ${s.mean_rating_other}`}
        </div>
      )}
      <table className="listen-log" data-testid="listening-log">
        <thead><tr><th>Fecha</th><th>A</th><th>B</th><th>Perfil</th><th>Prefiere</th><th>Nota A/B</th><th>Evaluador A/B</th><th>Comentario</th></tr></thead>
        <tbody>{(hist?.verdicts ?? []).map((v) => (
          <tr key={v.id} data-verdict={v.id}>
            <td>{v.created.slice(0, 19).replace("T", " ")}</td><td>{label(v, "a")}</td><td>{label(v, "b")}</td><td>{v.profile}</td>
            <td className="pref">{v.preferred === "none" ? "—" : v.preferred.toUpperCase()}</td>
            <td>{v.rating_a} / {v.rating_b}</td><td>{fmt(v.score_a)} / {fmt(v.score_b)}</td><td>{v.note}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}
