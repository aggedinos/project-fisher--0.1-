export type ProviderName = 'gemini' | 'ollama' | 'nvidia';
export type PermissionMode = 'auto' | 'safe' | 'supervised';
export type BrowserProfile = 'temporary' | 'persistent';

export interface ProviderOption {
  name: ProviderName;
  configured: boolean;
  models: string[];
}

export interface Settings {
  provider: ProviderName;
  model: string;
  permission_mode: PermissionMode;
  profile: BrowserProfile;
  headless: boolean;
  providers: ProviderOption[];
}

export interface TaskRequest {
  task: string;
  url?: string;
  provider: ProviderName;
  model?: string;
  permission_mode: PermissionMode;
  profile: BrowserProfile;
  headless: boolean;
}

export interface AgentEvent {
  type: string;
  task_id: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface PlanStep {
  id: string;
  description: string;
  done: boolean;
}

export interface TaskPlan {
  goal: string;
  steps: PlanStep[];
  current_step: number;
  revision: number;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface PermissionRequest {
  id: string;
  call: ToolCall;
  risk: 'low' | 'medium' | 'high';
  reason: string;
}

export interface Observation {
  url: string;
  title: string;
}

export interface SessionSummary {
  id: string;
  task: string;
  status: string;
  started_at?: string;
  ended_at?: string;
  provider?: string;
  answer?: string;
}

export interface SessionDetails extends SessionSummary {
  events: AgentEvent[];
}

export const defaultSettings: Settings = {
  provider: 'gemini',
  model: '',
  permission_mode: 'safe',
  profile: 'temporary',
  headless: false,
  providers: [
    { name: 'gemini', configured: false, models: [] },
    { name: 'ollama', configured: true, models: [] },
    { name: 'nvidia', configured: false, models: [] },
  ],
};
