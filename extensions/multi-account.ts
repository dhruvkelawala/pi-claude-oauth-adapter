/**
 * Multi-account Anthropic OAuth providers (`anthropic-2`, `anthropic-3`, …).
 *
 * Pi exposes Anthropic's OAuth implementation only through the built-in
 * provider descriptor; the `@earendil-works/pi-ai/oauth` subpath other
 * extensions import from is a type-only compatibility entry point whose
 * runtime module is empty (`export {}`). Extensions that import
 * `loginAnthropic` from it therefore fail with
 * `(0, _oauth.loginAnthropic) is not a function` the moment a user signs in.
 *
 * This module registers each configured extra subscription as its own
 * provider and delegates the OAuth flow to the built-in `anthropic`
 * descriptor resolved at runtime from Pi's registry, so login, refresh, and
 * credential storage behave exactly like the base account while each account
 * keeps its own credential entry (keyed by provider id).
 */
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/** Account config files, most specific first. Both use the same `subscriptions` shape. */
const ACCOUNT_CONFIG_FILES = ["claude-accounts.json", "multi-pass.json"];

export interface ClaudeAccount {
  readonly index: number;
  readonly label?: string;
}

/** Callback surface Pi hands to extension-registered OAuth providers. */
interface ExtensionOAuthCallbacks {
  onAuth(info: { url: string; instructions?: string }): void;
  onDeviceCode?(info: { userCode: string; verificationUri: string }): void;
  onPrompt(prompt: { message: string; placeholder?: string }): Promise<string>;
  onProgress?(message: string): void;
  onManualCodeInput?(): Promise<string>;
  onSelect?(prompt: { message: string; options: { id: string; label: string }[] }): Promise<string | undefined>;
  signal?: AbortSignal;
}

/** The built-in provider's OAuth descriptor (Pi's internal credential-based shape). */
interface BuiltinOAuthDescriptor {
  name?: string;
  isSubscription?: boolean;
  login(interaction: {
    signal?: AbortSignal;
    notify(event: { type: string; [key: string]: unknown }): void;
    prompt(prompt: { type: string; message?: string; placeholder?: string; options?: { id: string; label: string }[] }): Promise<string>;
  }): Promise<Record<string, unknown>>;
  refresh(credential: unknown, signal?: AbortSignal): Promise<Record<string, unknown>>;
}

interface RegistryLike {
  getProvider(provider: string): { baseUrl?: string; auth?: { oauth?: unknown } } | undefined;
  getAll(): Array<Record<string, unknown>>;
  registerProvider(name: string, config: Record<string, unknown>): void;
  unregisterProvider(name: string): void;
}

function agentDir(): string {
  const configured = process.env.PI_CODING_AGENT_DIR;
  return configured && configured.trim().length > 0 ? configured : join(homedir(), ".pi", "agent");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseAccount(value: unknown): ClaudeAccount | undefined {
  if (!isRecord(value) || value.provider !== "anthropic") return undefined;
  const index = value.index;
  if (typeof index !== "number" || !Number.isInteger(index) || index < 2) return undefined;
  const label = typeof value.label === "string" && value.label.trim() ? value.label.trim() : undefined;
  return label ? { index, label } : { index };
}

/** Read extra Claude subscriptions from the adapter config, falling back to pi-multi-pass's file. */
export function loadClaudeAccounts(directory: string = agentDir()): ClaudeAccount[] {
  for (const file of ACCOUNT_CONFIG_FILES) {
    const path = join(directory, file);
    if (!existsSync(path)) continue;
    try {
      const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
      if (!isRecord(parsed) || !Array.isArray(parsed.subscriptions)) continue;
      const accounts = parsed.subscriptions
        .map(parseAccount)
        .filter((account): account is ClaudeAccount => account !== undefined)
        .sort((left, right) => left.index - right.index);
      if (accounts.length > 0) return accounts;
    } catch {
      // Malformed config must never break session startup; try the next candidate.
    }
  }
  return [];
}

function isBuiltinOAuthDescriptor(value: unknown): value is BuiltinOAuthDescriptor {
  if (!isRecord(value)) return false;
  return typeof value.login === "function" && typeof value.refresh === "function";
}

/**
 * Bridge Pi's extension OAuth callback surface onto the built-in descriptor's
 * interaction surface so the account login renders the same prompts, URLs, and
 * progress messages as the base Anthropic provider.
 */
export function bridgeOAuth(builtin: BuiltinOAuthDescriptor, account: ClaudeAccount): Record<string, unknown> {
  const accountName = account.label ? `Anthropic — ${account.label}` : `Anthropic #${account.index}`;
  return {
    name: accountName,
    isSubscription: true,
    async login(callbacks: ExtensionOAuthCallbacks): Promise<Record<string, unknown>> {
      return builtin.login({
        signal: callbacks.signal,
        notify: (event) => {
          if (event.type === "auth_url" && typeof event.url === "string") {
            callbacks.onAuth({ url: event.url, instructions: typeof event.instructions === "string" ? event.instructions : undefined });
            return;
          }
          if (event.type === "device_code" && typeof event.userCode === "string" && typeof event.verificationUri === "string") {
            callbacks.onDeviceCode?.({ userCode: event.userCode, verificationUri: event.verificationUri });
            return;
          }
          if ((event.type === "progress" || event.type === "info") && typeof event.message === "string") {
            callbacks.onProgress?.(event.message);
          }
        },
        prompt: async (prompt) => {
          if (prompt.type === "select" && Array.isArray(prompt.options) && callbacks.onSelect) {
            const selected = await callbacks.onSelect({
              message: prompt.message ?? "",
              options: prompt.options.map((option) => ({ id: option.id, label: option.label })),
            });
            if (selected === undefined) throw new Error("Login cancelled");
            return selected;
          }
          if (prompt.type === "manual_code" && callbacks.onManualCodeInput) return callbacks.onManualCodeInput();
          return callbacks.onPrompt({ message: prompt.message ?? "", placeholder: prompt.placeholder });
        },
      });
    },
    async refreshToken(credentials: unknown, signal?: AbortSignal): Promise<Record<string, unknown>> {
      return builtin.refresh(credentials, signal);
    },
    // Pi requires a synchronous accessor here; the built-in `toAuth` is async,
    // and for Anthropic OAuth the access token is the request API key.
    getApiKey(credentials: unknown): string {
      return isRecord(credentials) && typeof credentials.access === "string" ? credentials.access : "";
    },
  };
}

function cloneModelsForAccount(registry: RegistryLike): Array<Record<string, unknown>> {
  return registry
    .getAll()
    .filter((model) => model.provider === "anthropic")
    .map((model) => ({
      id: model.id,
      name: model.name,
      api: model.api,
      baseUrl: model.baseUrl,
      reasoning: model.reasoning,
      thinkingLevelMap: model.thinkingLevelMap,
      input: model.input,
      cost: model.cost,
      contextWindow: model.contextWindow,
      maxTokens: model.maxTokens,
      compat: model.compat,
    }));
}

export interface RegisterAccountsResult {
  readonly registered: string[];
  readonly skipped?: string;
}

/**
 * Register every configured extra Claude subscription as `anthropic-N`.
 *
 * Runs on session start rather than activation so it lands after any other
 * extension that registers the same provider ids (notably pi-multi-pass, whose
 * Anthropic OAuth implementation is broken on this Pi line); the last
 * registration wins.
 *
 * `streamSimple` is the adapter's OAuth-aware Anthropic stream wrapper. Passing
 * it here gives extra accounts the same quota preflight and 429 translation the
 * base `anthropic` provider gets; without it, `anthropic-N` traffic would fall
 * back to Pi's raw streamer and silently skip those protections.
 */
export function registerAnthropicAccounts(registry: RegistryLike, streamSimple?: unknown): RegisterAccountsResult {
  const accounts = loadClaudeAccounts();
  if (accounts.length === 0) return { registered: [] };

  const base = registry.getProvider("anthropic");
  const builtinOAuth = base?.auth?.oauth;
  if (!isBuiltinOAuthDescriptor(builtinOAuth)) {
    return { registered: [], skipped: "built-in anthropic OAuth descriptor unavailable" };
  }

  const models = cloneModelsForAccount(registry);
  if (models.length === 0) return { registered: [], skipped: "no anthropic models available to clone" };

  const registered: string[] = [];
  for (const account of accounts) {
    const name = `anthropic-${account.index}`;
    try {
      registry.unregisterProvider(name);
    } catch {
      // Not previously registered (or owned by another extension); registration below still wins.
    }
    const config: Record<string, unknown> = {
      name: account.label ? `Claude (${account.label})` : `Claude #${account.index}`,
      baseUrl: base?.baseUrl,
      api: models[0]?.api,
      models,
      oauth: bridgeOAuth(builtinOAuth, account),
    };
    if (typeof streamSimple === "function") config.streamSimple = streamSimple;
    registry.registerProvider(name, config);
    registered.push(name);
  }
  return { registered };
}
