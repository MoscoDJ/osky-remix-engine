// Cliente del API local. Toda modificación es un lote de operaciones atómico (autor: usuario).

export interface MidiNote { tick: number; duration: number; pitch: number; velocity: number }
export interface AudioClip {
  type: "audio"; id: string; asset: string; start_tick: number;
  source_offset: number; source_length: number; gain_db: number;
}
export interface MidiClip { type: "midi"; id: string; start_tick: number; length_ticks: number; notes: MidiNote[] }
export type Clip = AudioClip | MidiClip;
export interface Plugin { id: string; backend: string; uri: string; params: Record<string, number>; bypass: boolean }
export interface Track {
  id: string; name: string; kind: string; clips: Clip[]; plugins: Plugin[];
  gain_db: number; pan: number; mute: boolean; solo: boolean; locked: boolean; output: string | null;
}
export interface Marker { id: string; tick: number; label: string }
export interface Project {
  name: string; sample_rate: number; ppq: number; key: string | null; master_profile: string;
  tempo: { tick: number; bpm: number }[]; tracks: Track[]; markers: Marker[];
}
export interface State {
  revision: string; project: Project; total_ticks: number; bar_ticks: number;
  audio_assets: Record<string, { name: string; frames: number; sample_rate: number }>;
  audio_clip_ticks: Record<string, number>;
}
export interface Revision { id: string; author: string; message: string; kind: string; created_at: string; ops: unknown[] }
export interface Job { id: string; status: string; result: any; error: string | null }

async function j<T>(r: Response): Promise<T> {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(body.reason || body.detail || r.statusText), { body, status: r.status });
  return body as T;
}

export const api = {
  projects: () => fetch("/api/projects").then(j<string[]>),
  info: (p: string) => fetch(`/api/projects/${p}`).then(j<{ branches: Record<string, string>; current: string; head: string }>),
  state: (p: string, rev?: string) => fetch(`/api/projects/${p}/state${rev ? `?rev=${encodeURIComponent(rev)}` : ""}`).then(j<State>),
  log: (p: string, branch: string) => fetch(`/api/projects/${p}/log?branch=${encodeURIComponent(branch)}&limit=40`).then(j<Revision[]>),
  profiles: () => fetch("/api/profiles").then(j<string[]>),
  ops: (p: string, ops: object[], message: string, branch: string) =>
    fetch(`/api/projects/${p}/ops`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ ops, message, branch }) }).then(j<{ ok: boolean; revision: string }>),
  branch: (p: string, name: string, from_rev: string) =>
    fetch(`/api/projects/${p}/branches`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, from_rev }) }).then(j<{ branch: string }>),
  checkout: (p: string, ref: string) =>
    fetch(`/api/projects/${p}/checkout`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ ref }) }).then(j<{ current: string }>),
  revert: (p: string, ref: string) =>
    fetch(`/api/projects/${p}/revert`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ ref }) }).then(j<{ revision: string }>),
  peaks: (p: string, sha: string) => fetch(`/api/projects/${p}/assets/${sha}/peaks?n=2000`).then(j<{ peaks: number[]; hop: number; frames: number }>),
  job: (p: string, body: object) =>
    fetch(`/api/projects/${p}/jobs`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body) }).then(j<{ job: string }>),
  getJob: (id: string) => fetch(`/api/jobs/${id}`).then(j<Job>),
};

/** Espera un trabajo: por WebSocket si llega el evento, con sondeo de respaldo. */
export async function runJob(p: string, body: object): Promise<any> {
  const { job } = await api.job(p, body);
  for (;;) {
    const s = await api.getJob(job);
    if (s.status === "done") return s.result;
    if (s.status === "failed") throw new Error(s.error || "falló el trabajo");
    await new Promise((r) => setTimeout(r, 400));
  }
}

export function connectEvents(onEvent: (e: any) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  const open = () => {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onclose = () => { if (!closed) setTimeout(open, 1000); };
  };
  open();
  return () => { closed = true; ws?.close(); };
}

export const secondsAt = (tick: number, prj: Project) => (tick / prj.ppq) * 60 / prj.tempo[0].bpm;
