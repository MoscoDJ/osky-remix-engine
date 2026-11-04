import { useCallback, useEffect, useMemo, useState } from "react";
import { api, Clip, connectEvents, MidiClip, MidiNote, Revision, runJob, State, Track } from "./api";
import ABPanel from "./components/ABPanel";
import History from "./components/History";
import Player from "./components/Player";
import MidiEditor from "./editor/MidiEditor";
import Mixer from "./mixer/Mixer";
import Timeline from "./timeline/Timeline";

type Tab = "mixer" | "midi" | "ab" | "eval";

export default function App() {
  const [projects, setProjects] = useState<string[]>([]);
  const [project, setProject] = useState<string>("");
  const [branches, setBranches] = useState<Record<string, string>>({});
  const [branch, setBranch] = useState("main");
  const [state, setState] = useState<State | null>(null);
  const [log, setLog] = useState<Revision[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("mixer");
  const [profiles, setProfiles] = useState<string[]>([]);
  const [profile, setProfile] = useState("");
  const [render, setRender] = useState<any>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [toast, setToast] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [evalRes, setEvalRes] = useState<any>(null);

  useEffect(() => {
    api.projects().then((ps) => { setProjects(ps); if (ps.length) setProject((p) => p || ps[0]); });
    api.profiles().then(setProfiles);
  }, []);

  const reload = useCallback(async () => {
    if (!project) return;
    const info = await api.info(project);
    setBranches(info.branches);
    const br = info.branches[branch] ? branch : info.current;
    if (br !== branch) setBranch(br);
    const [st, lg] = await Promise.all([api.state(project, br), api.log(project, br)]);
    setState(st);
    setLog(lg);
  }, [project, branch]);

  useEffect(() => { reload(); setRender(null); setEvalRes(null); }, [reload]);
  useEffect(() => connectEvents((e) => {
    if (e.project === project && (e.type === "revision" || e.type === "branch")) reload();
  }), [project, reload]);
  useEffect(() => { if (toast) { const id = setTimeout(() => setToast(null), 4500); return () => clearTimeout(id); } }, [toast]);

  const commit = async (ops: object[], message: string) => {
    try {
      await api.ops(project, ops, message, branch, state?.revision);
      setToast({ kind: "ok", text: message });
      await reload();
    } catch (e: any) {
      const b = e.body;
      setToast({ kind: "err", text: b?.index !== undefined ? `Operación #${b.index}: ${b.reason}. No se aplicó nada.` : e.message });
    }
  };

  const visibleRender = render?.revision === state?.revision && render?.requestedProfile === profile ? render : null;
  const prj = state?.project ?? null;
  const tracks = useMemo(() => prj?.tracks ?? [], [prj]);
  const selClip = useMemo(() => tracks.flatMap((t) => t.clips.map((c) => ({ t, c }))).find((x) => x.c.id === selected), [tracks, selected]);

  const onMove = (c: Clip, tick: number) => commit([{ op: "move_clip", clip: c.id, start_tick: tick }], `Mover ${c.id} al tick ${tick}`);
  const onLock = (t: Track) => commit([{ op: t.locked ? "unlock_track" : "lock_track", track: t.id }],
                                      `${t.locked ? "Desbloquear" : "Bloquear"} ${t.id}`);
  const onMix = (t: Track, upd: object) => commit([{ op: "set_track", track: t.id, ...upd }],
                                                  `Mezcla ${t.id}: ${Object.entries(upd).map(([k, v]) => `${k}=${v}`).join(", ")}`);
  const onSaveNotes = (notes: MidiNote[]) => selClip &&
    commit([{ op: "set_midi_notes", clip: selClip.c.id, notes }], `Editar notas de ${selClip.c.id} (${notes.length})`);

  const doRender = async () => {
    if (!state) return;
    setBusy("Renderizando…");
    try { const result = await runJob(project, { kind: "render", rev: state.revision, profile: profile || undefined }, setJobId); setRender({ ...result, requestedProfile: profile }); }
    catch (e: any) { setToast({ kind: "err", text: e.message }); }
    finally { setBusy(null); setJobId(null); }
  };
  const doEval = async () => {
    if (!state) return;
    setBusy("Evaluando…"); setTab("eval");
    try { setEvalRes(await runJob(project, { kind: "evaluate", rev: state.revision, profile: profile || undefined }, setJobId)); }
    catch (e: any) { setToast({ kind: "err", text: e.message }); }
    finally { setBusy(null); setJobId(null); }
  };
  const newBranch = async () => {
    const name = window.prompt("Nombre de la rama nueva", `${branch}-idea`);
    if (!name) return;
    try { await api.branch(project, name, branch); setBranch(name); }
    catch (e: any) { setToast({ kind: "err", text: e.message }); }
  };

  return (
    <div className="app">
      <header>
        <b className="logo">Osky Remix Engine</b>
        <select value={project} onChange={(e) => setProject(e.target.value)} aria-label="proyecto">
          {projects.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <select value={branch} onChange={(e) => setBranch(e.target.value)} aria-label="rama">
          {Object.keys(branches).map((b) => <option key={b} value={b}>{b}</option>)}
        </select>
        <button onClick={newBranch}>+ rama</button>
        <span className="sep" />
        {prj && <span className="meta">{prj.tempo[0].bpm} BPM · {prj.key ?? "—"} · {prj.sample_rate / 1000} kHz</span>}
        <select value={profile} onChange={(e) => setProfile(e.target.value)} aria-label="perfil">
          <option value="">perfil del proyecto ({prj?.master_profile})</option>
          {profiles.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <button className="primary" onClick={doRender} disabled={!!busy || !state}>Renderizar</button>
        <button onClick={doEval} disabled={!!busy || !state}>Evaluar</button>
        {busy && jobId && <button onClick={() => api.cancelJob(jobId).catch((e) => setToast({kind:"err", text:e.message}))}>Cancelar</button>}
        {busy && <span className="busy">{busy}</span>}
      </header>
      <main>
        <section className="left">
          <Player url={visibleRender?.master_url ?? null} prj={prj}
                  info={visibleRender ? `${visibleRender.lufs ?? "—"} LUFS · ${visibleRender.true_peak_dbtp ?? "—"} dBTP · calidad ${visibleRender.quality.status} · mastering ${visibleRender.mastering_status}` : undefined} />
          {state && <Timeline project={project} state={state} selected={selected} onSelect={(id) => { setSelected(id); if (id) setTab("midi"); }}
                              onMove={onMove} onToggleLock={onLock} />}
          <nav className="tabs">
            {(["mixer", "midi", "ab", "eval"] as Tab[]).map((x) => (
              <button key={x} className={tab === x ? "on" : ""} onClick={() => setTab(x)}>
                {{ mixer: "Mezclador", midi: "Editor MIDI", ab: "A/B", eval: "Evaluación" }[x]}
              </button>
            ))}
          </nav>
          <div className="panel">
            {tab === "mixer" && <Mixer tracks={tracks} onChange={onMix} onToggleLock={onLock} />}
            {tab === "midi" && (selClip && selClip.c.type === "midi"
              ? <MidiEditor clip={selClip.c as MidiClip} locked={selClip.t.locked} onSave={onSaveNotes} />
              : <div className="empty">Selecciona un clip MIDI en el timeline.</div>)}
            {tab === "ab" && <ABPanel project={project} branches={Object.keys(branches)} prj={prj} profile={profile || "club"} />}
            {tab === "eval" && (evalRes ? (
              <div className="eval" data-testid="eval">
                <div>Puntaje <b>{evalRes.score.toFixed(2)}</b> {evalRes.score === 0 && "· sin hallazgos"}</div>
                <table><tbody>{evalRes.metrics.sections.map((s: any) => <tr key={s.name}><td>{s.name}</td><td>energía {s.energy}</td><td>{s.lufs.toFixed(2)} LUFS</td></tr>)}</tbody></table>
                <ul>{evalRes.findings.map((f: any, i: number) => <li key={i}>{f.detail}</li>)}</ul>
              </div>) : <div className="empty">Pulsa «Evaluar».</div>)}
          </div>
        </section>
        <aside>
          <h3>Historial · {branch} · volver aquí revierte esta rama</h3>
          <History log={log} onRevert={(r) => state && api.revert(project, r.id, branch, state.revision).then(reload).catch((e) => setToast({ kind: "err", text: e.message }))} />
        </aside>
      </main>
      {toast && <div className={`toast ${toast.kind}`} role="status">{toast.text}</div>}
    </div>
  );
}
