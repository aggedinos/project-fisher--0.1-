import type { AgentEvent, SessionDetails, SessionSummary, Settings, TaskRequest } from './types';

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export const apiUrl = (path: string): string => `${apiBase}${path}`;

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json() as { detail?: string | { msg?: string }[]; message?: string };
      if (typeof body.detail === 'string') message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map(item => item.msg ?? '').join(', ');
      else if (body.message) message = body.message;
    } catch { /* The status code is still useful if there is no JSON body. */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Record<string, unknown>>('/api/health'),
  settings: () => request<Settings>('/api/settings'),
  saveSettings: (settings: Omit<Settings, 'providers'>) => request<Settings>('/api/settings', {
    method: 'PUT', body: JSON.stringify(settings),
  }),
  startTask: (body: TaskRequest) => request<{ task_id: string }>('/api/tasks', {
    method: 'POST', body: JSON.stringify(body),
  }),
  stopTask: (id: string) => request<Record<string, unknown>>(`/api/tasks/${encodeURIComponent(id)}/stop`, {
    method: 'POST',
  }),
  answerPermission: (id: string, requestId: string, approved: boolean) =>
    request<Record<string, unknown>>(`/api/tasks/${encodeURIComponent(id)}/permissions/${encodeURIComponent(requestId)}`, {
      method: 'POST', body: JSON.stringify({ approved }),
    }),
  sessions: async (): Promise<SessionSummary[]> => {
    const response = await request<SessionSummary[] | { sessions: SessionSummary[] }>('/api/sessions');
    return Array.isArray(response) ? response : response.sessions ?? [];
  },
  session: (id: string) => request<SessionDetails>(`/api/sessions/${encodeURIComponent(id)}`),
};

const namedEvents = [
  'agent_started', 'plan_updated', 'observation_updated', 'tool_started', 'tool_finished',
  'verification_result', 'permission_requested', 'frame', 'status', 'agent_finished', 'agent_error',
];

export function connectTaskEvents(
  id: string,
  onEvent: (event: AgentEvent) => void,
  onConnectionError: () => void,
): () => void {
  const stream = new EventSource(apiUrl(`/api/tasks/${encodeURIComponent(id)}/events`));
  const handle = (message: MessageEvent<string>) => {
    try {
      const event = JSON.parse(message.data) as AgentEvent;
      if (event && typeof event.type === 'string') onEvent(event);
    } catch { /* Ignore malformed or keepalive messages; the stream can continue. */ }
  };
  stream.onmessage = handle;
  namedEvents.forEach(name => stream.addEventListener(name, handle as EventListener));
  stream.onerror = onConnectionError;
  return () => stream.close();
}
