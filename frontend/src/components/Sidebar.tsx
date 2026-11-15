import type { Level, Profile, Risk, Status } from "../types";
import { Icon } from "./icons";
import { Segmented } from "./ui";

const LEVELS: Level[] = ["beginner", "intermediate", "advanced"];
const RISKS: Risk[] = ["conservative", "moderate", "aggressive"];

export function Sidebar({
  profile,
  setProfile,
  status,
  onNewChat,
}: {
  profile: Profile;
  setProfile: (p: Profile) => void;
  status: Status | null;
  onNewChat: () => void;
}) {
  return (
    <aside className="sidebar">
      <button className="btn primary" onClick={onNewChat}>
        <Icon name="plus" /> New chat
      </button>

      <div className="side-section">
        <div className="side-title">Your profile</div>
        <label className="field">
          Knowledge level
          <Segmented label="Knowledge level" options={LEVELS} value={profile.knowledge_level}
            onChange={(v) => setProfile({ ...profile, knowledge_level: v })} />
        </label>
        <label className="field">
          Risk appetite
          <Segmented label="Risk appetite" options={RISKS} value={profile.risk_tolerance}
            onChange={(v) => setProfile({ ...profile, risk_tolerance: v })} />
        </label>
        <label className="field">
          <span style={{ display: "flex", justifyContent: "space-between" }}>
            Investment horizon <b className="num" style={{ color: "var(--text)" }}>{profile.horizon_years} yrs</b>
          </span>
          <input type="range" min={1} max={40} value={profile.horizon_years}
            onChange={(e) => setProfile({ ...profile, horizon_years: Number(e.target.value) })} />
        </label>
      </div>

      <div className="side-section">
        <div className="side-title">Data connections</div>
        <ul className="status-list">
          {status
            ? Object.entries(status.services).map(([name, on]) => (
                <li key={name}>
                  <span>{name}</span>
                  <span className={`state ${on ? "on" : ""}`}>{on ? "● Live" : "○ Fallback"}</span>
                </li>
              ))
            : <li className="caption">Connecting…</li>}
        </ul>
      </div>

      <div className="caption" style={{ marginTop: "auto", display: "flex", gap: 8 }}>
        <Icon name="info" size={14} />
        <span>{status?.disclaimer ?? "Educational information only — not financial advice."}</span>
      </div>
    </aside>
  );
}
