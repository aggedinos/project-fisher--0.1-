import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity, AlertCircle, ArrowLeft, ArrowRight, Check, CheckCircle2, ChevronRight,
  Circle, Compass, FileClock, Globe2, History, LoaderCircle, Monitor,
  Pause, Play, Radio, RotateCcw, Save, Settings2, Shield, ShieldAlert, Square,
  Terminal, X,
} from 'lucide-react';
import { api, apiUrl, connectTaskEvents } from './api';
import type {
  AgentEvent, Observation, PermissionRequest, PlanStep, ProviderName, SessionDetails,
  SessionSummary, Settings, TaskPlan, ToolCall,
} from './types';
import { defaultSettings } from './types';

type RunState = 'idle' | 'running' | 'stopping' | 'completed' | 'stopped' | 'partial' | 'blocked' | 'error';
type View = 'monitor' | 'history';

const providerLabels: Record<ProviderName, string> = {
  gemini: 'Gemini', ollama: 'Ollama', nvidia: 'NVIDIA',
};

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : {};
}

function string(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function formatTime(value: string | undefined): string {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDate(value: string | undefined): string {
  if (!value) return 'Recent';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Recent' : date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function humanize(value: string): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
}

function getPlan(value: unknown): TaskPlan | null {
  const source = record(value);
  const steps = Array.isArray(source.steps) ? source.steps.map((step, index): PlanStep => {
    if (typeof step === 'string') return { id: String(index + 1), description: step, done: false };
    const item = record(step);
    return {
      id: string(item.id, String(index + 1)),
      description: string(item.description, string(item.title, `Step ${index + 1}`)),
      done: item.done === true,
    };
  }) : [];
  if (!steps.length && !source.goal) return null;
  return {
    goal: string(source.goal),
    steps,
    current_step: typeof source.current_step === 'number' ? source.current_step : 0,
    revision: typeof source.revision === 'number' ? source.revision : 0,
  };
}

function getCall(value: unknown): ToolCall | null {
  const source = record(value);
  return typeof source.name === 'string'
    ? { name: source.name, arguments: record(source.arguments) } : null;
}

function getPermission(value: unknown): PermissionRequest | null {
  const source = record(value);
  const call = getCall(source.call);
  return typeof source.id === 'string' && call
    ? {
      id: source.id,
      call,
      risk: source.risk === 'high' || source.risk === 'medium' ? source.risk : 'low',
      reason: string(source.reason),
    }
    : null;
}

function eventLabel(event: AgentEvent): string {
  const data = record(event.data);
  switch (event.type) {
    case 'agent_started': return 'Task started';
    case 'plan_updated': return 'Plan updated';
    case 'observation_updated': {
      const observation = record(data.observation);
      return string(observation.title, 'Page observed');
    }
    case 'tool_started': return `Running ${humanize(string(record(data.call).name, 'tool'))}`;
    case 'tool_finished': {
      const result = record(data.result);
      const call = record(result.call);
      return `${humanize(string(call.name, 'Tool'))} ${result.success === false ? 'failed' : 'finished'}`;
    }
    case 'verification_result': return data.changed === false ? 'No page change detected' : 'Action verified';
    case 'permission_requested': return 'Approval required';
    case 'agent_finished': return 'Task finished';
    case 'agent_error': return 'Task error';
    case 'status': return string(data.message, 'Status updated');
    default: return humanize(event.type);
  }
}

function eventDetail(event: AgentEvent): string {
  const data = record(event.data);
  switch (event.type) {
    case 'observation_updated': return string(record(data.observation).url);
    case 'tool_finished': return string(record(data.result).message);
    case 'verification_result': {
      const evidence = data.evidence;
      return Array.isArray(evidence) ? evidence.filter(item => typeof item === 'string').join(' · ') : '';
    }
    case 'agent_finished': {
      const result = record(data.result);
      return string(result.answer, string(result.status));
    }
    case 'agent_error': return string(data.message);
    default: return '';
  }
}

function eventKind(event: AgentEvent): 'good' | 'warn' | 'neutral' {
  if (event.type === 'agent_error' || event.type === 'permission_requested') return 'warn';
  if (event.type === 'agent_finished' || event.type === 'verification_result') return 'good';
  return 'neutral';
}

function EventTimeline({ events, compact = false }: { events: AgentEvent[]; compact?: boolean }) {
  const visible = compact ? events.filter(event => event.type !== 'frame').slice(-35).reverse() : events.filter(event => event.type !== 'frame');
  if (!visible.length) return <div className="empty-events"><Activity size={22} /><p>Activity will appear as the agent works.</p></div>;
  return <div className={`event-timeline ${compact ? 'compact' : ''}`}>
    {visible.map((event, index) => <div className="timeline-item" key={`${event.timestamp}-${event.type}-${index}`}>
      <span className={`timeline-dot ${eventKind(event)}`} />
      <div className="timeline-copy">
        <div className="timeline-title">{eventLabel(event)}</div>
        {eventDetail(event) && <div className="timeline-detail">{eventDetail(event)}</div>}
      </div>
      <time>{formatTime(event.timestamp)}</time>
    </div>)}
  </div>;
}

function SettingsDialog({
  initial, onClose, onSaved,
}: { initial: Settings; onClose: () => void; onSaved: (settings: Settings) => void }) {
  const [draft, setDraft] = useState(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const selectedProvider = draft.providers.find(provider => provider.name === draft.provider);
  const patch = (changes: Partial<Settings>) => setDraft(current => ({ ...current, ...changes }));
  const save = async () => {
    setSaving(true); setError('');
    try {
      const result = await api.saveSettings({
        provider: draft.provider,
        model: draft.model.trim(),
        permission_mode: draft.permission_mode,
        profile: draft.profile,
        headless: draft.headless,
      });
      onSaved({ ...draft, ...result, providers: result.providers ?? draft.providers });
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not save settings'); }
    finally { setSaving(false); }
  };
  return <div className="modal-backdrop" onMouseDown={(event: { target: EventTarget; currentTarget: EventTarget }) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <header className="dialog-header">
        <div><span className="eyebrow">PREFERENCES</span><h2 id="settings-title">Agent settings</h2><p className="field-help">Saved on this device. No Fisher account is needed.</p></div>
        <button className="icon-button" onClick={onClose} aria-label="Close settings"><X size={18} /></button>
      </header>
      <div className="dialog-body">
        <div className="settings-group">
          <div className="settings-group-title"><Radio size={17} /><span>Model provider</span></div>
          <div className="provider-grid">
            {draft.providers.map(provider => <button
              type="button" key={provider.name}
              className={`provider-card ${draft.provider === provider.name ? 'selected' : ''}`}
              onClick={() => patch({ provider: provider.name, model: provider.models[0] ?? '' })}
            >
              <span>{providerLabels[provider.name] ?? provider.name}</span>
              <small className={provider.configured ? 'available' : 'unavailable'}>{provider.name === 'ollama' ? 'Local · no account' : provider.configured ? 'API key configured' : 'API key needed'}</small>
            </button>)}
          </div>
          <label className="field-label" htmlFor="model-input">Model</label>
          <input id="model-input" className="text-input" value={draft.model} list="provider-models"
            onChange={(event: { target: HTMLInputElement }) => patch({ model: event.target.value })} placeholder="Provider default" />
          <datalist id="provider-models">{selectedProvider?.models.map(model => <option value={model} key={model} />)}</datalist>
          <p className="field-help">Choose a listed model or enter one supported by your provider.</p>
        </div>
        <div className="settings-group">
          <div className="settings-group-title"><Shield size={17} /><span>Permission mode</span></div>
          <div className="segmented-options">
            {(['auto', 'safe', 'supervised'] as const).map(mode => <button type="button" key={mode}
              className={draft.permission_mode === mode ? 'active' : ''}
              onClick={() => patch({ permission_mode: mode })}>{humanize(mode)}</button>)}
          </div>
          <p className="field-help">Safe asks before medium and high risk actions. Supervised asks before every meaningful action.</p>
        </div>
        <div className="settings-group last">
          <div className="settings-group-title"><Monitor size={17} /><span>Browser</span></div>
          <label className="field-label" htmlFor="profile-select">Profile</label>
          <select id="profile-select" className="text-input" value={draft.profile}
            onChange={(event: { target: HTMLSelectElement }) => patch({ profile: event.target.value as Settings['profile'] })}>
            <option value="temporary">Temporary, isolated</option>
            <option value="persistent">Persistent Fisher profile</option>
          </select>
          <label className="toggle-row" htmlFor="headless-toggle">
            <span><strong>Headless browser</strong><small>Run without a visible browser window</small></span>
            <input id="headless-toggle" type="checkbox" checked={draft.headless}
              onChange={(event: { target: HTMLInputElement }) => patch({ headless: event.target.checked })} />
            <span className="toggle-switch" aria-hidden="true" />
          </label>
        </div>
      </div>
      <footer className="dialog-footer">
        {error && <span className="inline-error"><AlertCircle size={15} />{error}</span>}
        <button className="secondary-button" onClick={onClose}>Cancel</button>
        <button className="primary-button" onClick={save} disabled={saving}>
          {saving ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />} Save settings
        </button>
      </footer>
    </section>
  </div>;
}

function ReplayPanel({ session, index, onIndex, playing, onPlay }: {
  session: SessionDetails; index: number; onIndex: (index: number) => void;
  playing: boolean; onPlay: () => void;
}) {
  const events = session.events.filter(event => event.type !== 'frame');
  const selected = events[index];
  return <div className="replay-panel">
    <div className="replay-heading"><div><span className="eyebrow">SESSION REPLAY</span><h2>{session.task || 'Untitled task'}</h2>
      <p>{formatDate(session.started_at)} · {events.length} recorded events · {humanize(session.status)}</p></div>
      <span className="session-status">{session.status}</span>
    </div>
    <div className="replay-stage">
      <div className="replay-orb"><FileClock size={38} strokeWidth={1.4} /></div>
      <div className="replay-step-label">EVENT {events.length ? index + 1 : 0} / {events.length}</div>
      <h3>{selected ? eventLabel(selected) : 'No recorded events'}</h3>
      <p>{selected ? eventDetail(selected) || 'The agent recorded this step while working.' : 'This session has no replayable events.'}</p>
      <div className="replay-controls">
        <button className="icon-button" aria-label="Previous event" disabled={index <= 0} onClick={() => onIndex(index - 1)}><ArrowLeft size={18} /></button>
        <button className="play-button" aria-label={playing ? 'Pause replay' : 'Play replay'} disabled={!events.length}
          onClick={onPlay}>{playing ? <Pause size={18} /> : <Play size={18} fill="currentColor" />}</button>
        <button className="icon-button" aria-label="Next event" disabled={index >= events.length - 1}
          onClick={() => onIndex(index + 1)}><ArrowRight size={18} /></button>
      </div>
      <input className="replay-range" type="range" min={0} max={Math.max(0, events.length - 1)} value={Math.min(index, Math.max(0, events.length - 1))}
        onChange={(event: { target: HTMLInputElement }) => onIndex(Number(event.target.value))} aria-label="Replay position" disabled={!events.length} />
    </div>
    <div className="replay-events"><div className="section-heading"><h3>Recorded activity</h3><span>{events.length} events</span></div>
      <EventTimeline events={events} /></div>
  </div>;
}

export default function App() {
  const [view, setView] = useState<View>('monitor');
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [task, setTask] = useState('');
  const [url, setUrl] = useState('');
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [runState, setRunState] = useState<RunState>('idle');
  const [statusText, setStatusText] = useState('Ready for a new task');
  const [error, setError] = useState('');
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [plan, setPlan] = useState<TaskPlan | null>(null);
  const [observation, setObservation] = useState<Observation | null>(null);
  const [currentCall, setCurrentCall] = useState<ToolCall | null>(null);
  const [permission, setPermission] = useState<PermissionRequest | null>(null);
  const [permissionBusy, setPermissionBusy] = useState(false);
  const [answer, setAnswer] = useState('');
  const [frameUrl, setFrameUrl] = useState('');
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSession, setSelectedSession] = useState<SessionDetails | null>(null);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [replayIndex, setReplayIndex] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);

  const isRunning = runState === 'running' || runState === 'stopping';
  const refreshSessions = useCallback(async () => {
    try { setSessions(await api.sessions()); } catch { /* The live task remains usable if history is unavailable. */ }
  }, []);

  useEffect(() => {
    void api.health().then(() => setConnected(true)).catch(() => setConnected(false));
    void api.settings().then(remote => setSettings({ ...defaultSettings, ...remote })).catch(() => {});
    void refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    if (!activeTaskId) return;
    const disconnect = connectTaskEvents(activeTaskId, event => {
      setEvents(current => [...current.slice(-199), event]);
      const data = record(event.data);
      switch (event.type) {
        case 'agent_started': setStatusText('Understanding task'); break;
        case 'plan_updated': setPlan(getPlan(data.plan)); setStatusText('Plan updated'); break;
        case 'observation_updated': {
          const page = record(data.observation);
          setObservation({ url: string(page.url), title: string(page.title) });
          setStatusText(string(page.title, 'Inspecting page'));
          break;
        }
        case 'tool_started': {
          const call = getCall(data.call);
          setCurrentCall(call);
          setStatusText(call ? `Running ${humanize(call.name)}` : 'Executing action');
          break;
        }
        case 'tool_finished': setCurrentCall(null); break;
        case 'verification_result': setStatusText(data.changed === false ? 'Checking another approach' : 'Action verified'); break;
        case 'permission_requested': {
          setPermission(getPermission(data.request));
          setStatusText('Waiting for your approval');
          break;
        }
        case 'status': setStatusText(string(data.message, 'Working')); break;
        case 'agent_finished': {
          const result = record(data.result);
          const outcome = string(result.status);
          const nextState: RunState = outcome === 'completed' ? 'completed'
            : outcome === 'stopped' || outcome === 'cancelled' ? 'stopped'
            : outcome === 'partial' ? 'partial'
            : outcome === 'blocked' ? 'blocked' : 'error';
          setRunState(nextState);
          setStatusText(nextState === 'completed' ? 'Task completed'
            : nextState === 'stopped' ? 'Task stopped'
            : nextState === 'partial' ? 'Task partially completed'
            : nextState === 'blocked' ? 'Task blocked' : 'Task failed');
          if (typeof result.error === 'string' && result.error) setError(result.error);
          setAnswer(string(result.answer));
          setCurrentCall(null);
          setPermission(null);
          disconnect();
          void refreshSessions();
          break;
        }
        case 'agent_error': {
          setRunState('error');
          setStatusText('Task failed');
          setError(string(data.message, 'The agent encountered an error.'));
          setCurrentCall(null);
          setPermission(null);
          disconnect();
          void refreshSessions();
          break;
        }
      }
    }, () => setStatusText(current => current === 'Task completed' ? current : 'Reconnecting to agent…'));
    return disconnect;
  }, [activeTaskId, refreshSessions]);

  useEffect(() => {
    if (!activeTaskId || !isRunning) return;
    let cancelled = false;
    let loading = false;
    const updateFrame = () => {
      if (loading) return;
      loading = true;
      const nextUrl = apiUrl(`/api/tasks/${encodeURIComponent(activeTaskId)}/frame?t=${Date.now()}`);
      const next = new Image();
      next.onload = () => { if (!cancelled) setFrameUrl(nextUrl); loading = false; };
      next.onerror = () => { loading = false; };
      next.src = nextUrl;
    };
    updateFrame();
    const timer = window.setInterval(updateFrame, 1600);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [activeTaskId, isRunning]);

  useEffect(() => {
    if (!replayPlaying || !selectedSession) return;
    const length = selectedSession.events.filter(event => event.type !== 'frame').length;
    const timer = window.setInterval(() => setReplayIndex(current => {
      if (current >= length - 1) { setReplayPlaying(false); return current; }
      return current + 1;
    }), 1100);
    return () => window.clearInterval(timer);
  }, [replayPlaying, selectedSession]);

  const startTask = async () => {
    const prompt = task.trim();
    if (!prompt || isRunning) return;
    setError('');
    try {
      const response = await api.startTask({
        task: prompt,
        ...(url.trim() ? { url: url.trim() } : {}),
        provider: settings.provider,
        ...(settings.model.trim() ? { model: settings.model.trim() } : {}),
        permission_mode: settings.permission_mode,
        profile: settings.profile,
        headless: settings.headless,
      });
      setActiveTaskId(response.task_id);
      setRunState('running');
      setStatusText('Starting browser');
      setEvents([]); setPlan(null); setObservation(null); setCurrentCall(null);
      setPermission(null); setAnswer(''); setFrameUrl(''); setView('monitor');
      setConnected(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not start task');
    }
  };

  const stopTask = async () => {
    if (!activeTaskId || !isRunning) return;
    setRunState('stopping'); setStatusText('Stopping task…'); setError('');
    try { await api.stopTask(activeTaskId); }
    catch (cause) {
      setRunState('running');
      setError(cause instanceof Error ? cause.message : 'Could not stop task');
    }
  };

  const answerPermission = async (approved: boolean) => {
    if (!activeTaskId || !permission || permissionBusy) return;
    setPermissionBusy(true); setError('');
    try {
      await api.answerPermission(activeTaskId, permission.id, approved);
      setPermission(null);
      setStatusText(approved ? 'Action approved' : 'Action declined');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not send decision'); }
    finally { setPermissionBusy(false); }
  };

  const openSession = async (session: SessionSummary) => {
    setView('history'); setSessionLoading(true); setReplayPlaying(false); setReplayIndex(0); setError('');
    try { setSelectedSession(await api.session(session.id)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not open session'); }
    finally { setSessionLoading(false); }
  };

  const progress = useMemo(() => {
    if (!plan?.steps.length) return 0;
    return Math.round(plan.steps.filter(step => step.done).length / plan.steps.length * 100);
  }, [plan]);

  return <div className="app-shell">
    <aside className="rail">
      <div className="brand-mark" title="Project Fisher"><Compass size={26} strokeWidth={1.9} /></div>
      <div className="rail-nav">
        <button className={`rail-button ${view === 'monitor' ? 'active' : ''}`} onClick={() => setView('monitor')} title="Monitor" aria-label="Monitor"><Monitor size={20} /></button>
        <button className={`rail-button ${view === 'history' ? 'active' : ''}`} onClick={() => { setView('history'); void refreshSessions(); }} title="Sessions" aria-label="Sessions"><History size={20} /></button>
      </div>
      <button className="rail-button rail-settings" onClick={() => setSettingsOpen(true)} title="Settings" aria-label="Settings"><Settings2 size={20} /></button>
    </aside>

    <div className="workspace">
      <header className="topbar">
        <div className="brand-name"><span>PROJECT</span><strong>FISHER</strong><span className="version">0.2</span></div>
        <div className="topbar-right">
          <span className={`connection-pill ${connected === true ? 'connected' : connected === false ? 'disconnected' : ''}`}>
            <span className="connection-dot" />{connected === true ? 'Connected' : connected === false ? 'Offline' : 'Connecting'}
          </span>
          <span className="topbar-divider" />
          <button className="topbar-settings" onClick={() => setSettingsOpen(true)}><Settings2 size={17} /><span>Settings</span></button>
        </div>
      </header>

      {error && <div className="error-banner" role="alert"><AlertCircle size={18} /><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}

      <div className="main-layout">
        <main className="main-content">
          {view === 'monitor' ? <>
            <div className="page-heading"><div><span className="eyebrow">AGENT WORKSPACE</span><h1>Control center<span className="heading-period">.</span></h1>
              <p>Give Fisher a goal. Follow every step as it works.</p></div>
              <div className={`run-pill ${runState}`}><span className="pulse-dot" />{isRunning ? runState === 'stopping' ? 'Stopping' : 'Agent active' : humanize(runState)}</div>
            </div>

            <section className="browser-card" aria-label="Live browser preview">
              <div className="browser-toolbar">
                <div className="browser-dots"><span /><span /><span /></div>
                <div className="address-bar"><Globe2 size={14} /><span title={observation?.url || url || ''}>{observation?.url || url || 'Browser preview'}</span></div>
                <span className={`live-indicator ${isRunning ? 'on' : ''}`}><span /> LIVE</span>
              </div>
              <div className={`browser-viewport ${frameUrl ? 'has-frame' : ''}`}>
                {frameUrl ? <img src={frameUrl} alt="Current browser page" /> : <div className="browser-placeholder">
                  <div className="preview-glow" /><div className="preview-icon"><Compass size={41} strokeWidth={1.2} /></div>
                  <h2>{isRunning ? 'Opening browser view' : 'Your browser, in view'}</h2>
                  <p>{isRunning ? 'The first frame will appear once the page is ready.' : 'Start a task to watch Fisher navigate in real time.'}</p>
                </div>}
              </div>
              <div className="browser-footer"><span><span className={`footer-status-dot ${isRunning ? 'active' : ''}`} />{statusText}</span><span>{observation?.title || 'No active page'}</span></div>
            </section>

            {permission && <section className="permission-card" role="alertdialog" aria-label="Permission request">
              <div className="permission-icon"><ShieldAlert size={22} /></div>
              <div className="permission-copy"><span className="eyebrow">YOUR APPROVAL IS REQUIRED · {permission.risk.toUpperCase()} RISK</span>
                <h3>{humanize(permission.call.name)}</h3><p>{permission.reason || 'Fisher needs approval before continuing with this action.'}</p></div>
              <div className="permission-actions"><button className="secondary-button" disabled={permissionBusy} onClick={() => void answerPermission(false)}>Decline</button>
                <button className="primary-button" disabled={permissionBusy} onClick={() => void answerPermission(true)}>{permissionBusy ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />} Approve</button></div>
            </section>}

            {answer && !isRunning && <section className="answer-card"><div className="answer-icon"><CheckCircle2 size={20} /></div><div><span className="eyebrow">TASK RESULT</span><p>{answer}</p></div></section>}

            <section className="composer-card" aria-label="Start a task">
              <div className="composer-top"><span className="eyebrow">NEW TASK</span><span className="keyboard-hint">Ctrl + Enter to run</span></div>
              <textarea aria-label="Task" placeholder="What would you like Fisher to do?" value={task} onChange={(event: { target: HTMLTextAreaElement }) => setTask(event.target.value)}
                onKeyDown={event => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); void startTask(); } }} rows={3} />
              <div className="composer-bottom"><div className="composer-url"><Globe2 size={16} /><input aria-label="Starting URL" type="url" placeholder="Starting URL (optional)" value={url} onChange={(event: { target: HTMLInputElement }) => setUrl(event.target.value)} /></div>
                <div className="composer-actions"><span className="composer-model"><Circle size={9} fill="currentColor" />{providerLabels[settings.provider]}{settings.model ? ` · ${settings.model}` : ''}</span>
                  {isRunning ? <button className="stop-button" onClick={() => void stopTask()} disabled={runState === 'stopping'}><Square size={15} fill="currentColor" /> Stop</button>
                    : <button className="run-button" onClick={() => void startTask()} disabled={!task.trim()}><Play size={16} fill="currentColor" /> Run task <ArrowRight size={16} /></button>}</div></div>
            </section>
          </> : <>
            <div className="page-heading"><div><span className="eyebrow">YOUR RECORDINGS</span><h1>Sessions<span className="heading-period">.</span></h1>
              <p>Review the agent’s recorded decisions and progress.</p></div><button className="subtle-button" onClick={() => void refreshSessions()}><RotateCcw size={15} /> Refresh</button></div>
            {sessionLoading ? <div className="history-placeholder"><LoaderCircle className="spin" size={26} /> Loading session…</div>
              : selectedSession ? <ReplayPanel session={selectedSession} index={replayIndex} onIndex={index => { setReplayIndex(index); setReplayPlaying(false); }}
                  playing={replayPlaying} onPlay={() => { if (replayIndex >= selectedSession.events.filter(event => event.type !== 'frame').length - 1) setReplayIndex(0); setReplayPlaying(!replayPlaying); }} />
              : <div className="history-placeholder"><FileClock size={35} strokeWidth={1.4} /><h2>Select a session</h2><p>Choose a recording from the list to replay its activity.</p></div>}
          </>}
        </main>

        <aside className="inspector">
          <section className="inspector-section overview-section">
            <div className="section-heading"><h2>Overview</h2><span className="inspector-number">01</span></div>
            <div className="overview-status"><div className={`overview-icon ${isRunning ? 'running' : ''}`}><Activity size={20} /></div>
              <div><span className="mini-label">CURRENT STATUS</span><strong>{statusText}</strong></div></div>
            <div className="overview-meta"><div><span>Provider</span><strong>{providerLabels[settings.provider]}</strong></div><div><span>Permission</span><strong>{humanize(settings.permission_mode)}</strong></div></div>
          </section>
          <section className="inspector-section plan-section">
            <div className="section-heading"><h2>Current plan</h2><span className="inspector-number">02</span></div>
            {plan && plan.steps.length ? <><div className="plan-progress"><div><span>Progress</span><strong>{progress}%</strong></div><div className="progress-track"><span style={{ width: `${progress}%` }} /></div></div>
              <div className="plan-list">{plan.steps.map((step, index) => <div className={`plan-step ${step.done ? 'done' : index === plan.current_step ? 'current' : ''}`} key={step.id}>
                <span className="plan-step-icon">{step.done ? <Check size={13} /> : index === plan.current_step ? <ChevronRight size={15} /> : <Circle size={13} />}</span>
                <span>{step.description}</span></div>)}</div></>
              : <div className="empty-plan"><Compass size={25} /><p>A plan will appear when the agent starts working.</p></div>}
          </section>
          <section className="inspector-section action-section">
            <div className="section-heading"><h2>Current action</h2><span className="inspector-number">03</span></div>
            <div className="action-box"><Terminal size={16} /><code>{currentCall ? `${currentCall.name}(…)` : isRunning ? 'Observing page…' : 'Waiting for task'}</code></div>
          </section>
          <section className="inspector-section activity-section">
            <div className="section-heading"><h2>Activity</h2><span className="activity-count">{events.filter(event => event.type !== 'frame').length}</span></div>
            <EventTimeline events={events} compact />
          </section>
          <section className="inspector-section sessions-section">
            <div className="section-heading"><h2>Recent sessions</h2><button className="text-button" onClick={() => { setView('history'); void refreshSessions(); }}>View all <ArrowRight size={13} /></button></div>
            {sessions.length ? <div className="session-list">{sessions.slice(0, 5).map(session => <button key={session.id} className={`session-item ${selectedSession?.id === session.id && view === 'history' ? 'selected' : ''}`} onClick={() => void openSession(session)}>
              <span className="session-icon"><FileClock size={16} /></span><span className="session-info"><strong>{session.task || 'Untitled task'}</strong><small>{formatDate(session.started_at)} · {humanize(session.status)}</small></span><ChevronRight size={15} /></button>)}</div>
              : <p className="empty-sessions">Completed runs will appear here.</p>}
          </section>
        </aside>
      </div>
      <footer className="app-footer"><span><Shield size={13} /> Local control center</span><span>Project Fisher 0.2</span></footer>
    </div>
    {settingsOpen && <SettingsDialog initial={settings} onClose={() => setSettingsOpen(false)} onSaved={saved => { setSettings(saved); setSettingsOpen(false); }} />}
  </div>;
}
