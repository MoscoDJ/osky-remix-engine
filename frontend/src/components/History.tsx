import { Revision } from "../api";

interface Props { log: Revision[]; onRevert: (r: Revision) => void }

export default function History({ log, onRevert }: Props) {
  return (
    <div className="history">
      {log.map((r, i) => (
        <div key={r.id} className={`rev ${r.author}`} data-rev={r.id}>
          <code>{r.id.slice(0, 8)}</code>
          <span className={`who ${r.author}`}>{r.author === "user" ? "tú" : r.author}</span>
          <span className="msg" title={r.message}>{r.message}</span>
          <span className="when">{r.created_at.slice(11, 19)}</span>
          {i > 0 && <button onClick={() => onRevert(r)} title="Crea una revisión nueva con este contenido">volver aquí</button>}
        </div>
      ))}
    </div>
  );
}
