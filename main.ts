// main.ts — the free-model aggregator server.
//
// Flow: read the brief, fetch the provider directory from the env, probe
// every provider's catalog route, register port + catalog with the env, then
// serve the env client's requests until /work says done, and exit 0.
//
// Routing: for each request, rank the free models that meet its requirements
// by sealed rating (ties: provider id, then model id) and call them in
// order. A refusal's message names its cause: a spent budget or an outage
// ("...'s free-tier budget is spent", "... is out") ends the whole provider
// at once, with no retry; a failure-schedule refusal ("... refused this
// call") is transient, so it is retried once on the same model, and only
// then is the model marked dead. A 400 (unknown or paid model) marks the
// model dead at once; a 401 marks the whole provider dead. Requests arrive
// in manifest order, weight-2 first, so the best models' small budgets
// naturally go to the heaviest requests first.
//
// Determinism: no wall clock and no unseeded randomness anywhere in a
// request path. Fixed ids and created values; the port comes from the OS.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import {
  parseBrief,
  buildCatalog,
  catalogFromBrief,
  rankCandidates,
  freeModelIds,
  RouterState,
  providerOf,
} from "./lib/providers.ts";
import type { Brief, Catalog, ProviderDir, Requires } from "./lib/providers.ts";
import {
  normalizeOpenAI,
  normalizeAnthropic,
  callProvider,
  openAIJsonResponse,
  openAISseResponse,
  anthropicJsonResponse,
  anthropicSseResponse,
} from "./lib/wire.ts";
import type { NormRequest } from "./lib/wire.ts";

const EMPTY_CATALOG: Catalog = { providers: [] };

function env(name: string): string | undefined {
  const value = process.env[name];
  return value === undefined || value === "" ? undefined : value;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/** Read and parse the brief, or an empty brief when there is none. */
function readBrief(): Brief {
  const file = env("MAKEFUN_PROMPT_FILE");
  if (file === undefined) return { directory: [], ratings: [], manifest: [] };
  try {
    return parseBrief(readFileSync(file, "utf8"));
  } catch {
    return { directory: [], ratings: [], manifest: [] };
  }
}

/**
 * Fetch the provider directory from the env's /instance route, which gives
 * absolute base URLs. Falls back to the brief's directory joined to the env
 * origin when /instance cannot be reached.
 */
async function loadDirectory(brief: Brief, envUrl: string): Promise<ProviderDir[]> {
  try {
    const res = await fetch(`${envUrl.replace(/\/$/, "")}/instance`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const body: unknown = await res.json();
      if (isRecord(body) && Array.isArray(body.providers)) {
        const out: ProviderDir[] = [];
        for (const raw of body.providers) {
          if (!isRecord(raw)) continue;
          const id = typeof raw.id === "string" ? raw.id : "";
          const known = brief.directory.find((p) => p.id === id);
          out.push({
            id,
            name: typeof raw.name === "string" ? raw.name : known?.name ?? id,
            wire: (raw.wire as ProviderDir["wire"]) ?? known?.wire ?? "openai",
            auth: (raw.auth as ProviderDir["auth"]) ?? known?.auth ?? "none",
            key_env: typeof raw.key_env === "string" ? raw.key_env : known?.key_env ?? "",
            key: typeof raw.key === "string" ? raw.key : known?.key ?? "",
            base_url: typeof raw.base_url === "string" ? raw.base_url : known?.base_url ?? "",
          });
        }
        if (out.length > 0) return out;
      }
    }
  } catch {
    // Fall through to the brief's directory.
  }
  const origin = envUrl.replace(/\/$/, "");
  return brief.directory.map((p) => ({ ...p, base_url: p.base_url.startsWith("http") ? p.base_url : `${origin}${p.base_url}` }));
}

/** Build the catalog: probe every provider, fall back to the brief. */
async function loadCatalog(brief: Brief): Promise<Catalog> {
  if (brief.directory.length === 0) return EMPTY_CATALOG;
  try {
    return await buildCatalog(brief);
  } catch {
    return catalogFromBrief(brief);
  }
}

/**
 * The brief the trial runs on: the directory from the env's /instance route
 * (absolute base URLs) when the env is reachable, else the brief's own
 * directory. The catalog probing and every provider call must use these
 * same base URLs — the brief's relative paths ("/p/groq") cannot be dialled.
 */
async function loadEffectiveBrief(brief: Brief, envUrl: string | undefined): Promise<Brief> {
  if (brief.directory.length === 0 || envUrl === undefined) return brief;
  const directory = await loadDirectory(brief, envUrl);
  return { ...brief, directory };
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

interface ServerState {
  brief: Brief;
  catalog: Catalog;
  router: RouterState;
  served: number;
}

function requirementsFor(state: ServerState, api: "openai" | "anthropic", norm: NormRequest): Requires {
  const entry = state.brief.manifest[state.served];
  if (entry !== undefined && entry.api === api) return entry.requires;
  // No manifest entry (or a mismatch): derive what the body itself needs.
  const req: Requires = {};
  if (norm.tools.length > 0) req.tools = true;
  if (norm.messages.some((m) => m.images.length > 0)) req.vision = true;
  if (norm.jsonMode) req.json = true;
  return req;
}

/** A refusal whose message says the provider's free-tier budget is spent. */
function budgetSpent(message: string): boolean {
  return message.endsWith("free-tier budget is spent");
}

/** A refusal whose message says the provider is in outage. */
function inOutage(message: string): boolean {
  return message.endsWith(" is out");
}

async function handleRequest(
  state: ServerState,
  api: "openai" | "anthropic",
  body: Record<string, unknown>,
): Promise<{ status: number; contentType: string; payload: string }> {
  const norm = api === "openai" ? normalizeOpenAI(body) : normalizeAnthropic(body);
  const stream = body.stream === true;
  const requires = requirementsFor(state, api, norm);
  const candidates = rankCandidates(state.brief, state.router, requires);
  state.served += 1;
  for (const candidate of candidates) {
    const provider = providerOf(state.brief, candidate.provider);
    if (provider === undefined) continue;
    const attempt = async () => callProvider(provider, candidate.model, norm);
    let result = await attempt();
    if (!result.ok) {
      const failure = result.failure;
      if (failure.status === 400) {
        // Unknown or paid model: never pick it again, no retry.
        state.router.markModelDead(candidate.provider, candidate.model);
        continue;
      }
      if (failure.status === 401) {
        // Wrong key: the whole provider is unusable.
        state.router.markProviderDead(candidate.provider);
        continue;
      }
      if (budgetSpent(failure.message) || inOutage(failure.message)) {
        // A spent budget or an outage ends the provider for this trial:
        // no retry, and no other model on it either.
        state.router.markProviderDead(candidate.provider);
        continue;
      }
      if (failure.status === 402 || failure.status === 429 || failure.status >= 500 || failure.status === 0) {
        // A failure-schedule refusal is transient: retry once on the
        // same model before giving up on it.
        result = await attempt();
        if (!result.ok) {
          if (budgetSpent(result.failure.message) || inOutage(result.failure.message)) {
            state.router.markProviderDead(candidate.provider);
          } else {
            state.router.markModelDead(candidate.provider, candidate.model);
          }
          continue;
        }
      } else {
        state.router.markModelDead(candidate.provider, candidate.model);
        continue;
      }
    }
    if (result.ok) {
      const model = candidate.model;
      if (api === "openai") {
        if (stream) return { status: 200, contentType: "text/event-stream; charset=utf-8", payload: openAISseResponse(model, result.answer) };
        return { status: 200, contentType: "application/json", payload: openAIJsonResponse(model, result.answer) };
      }
      if (stream) return { status: 200, contentType: "text/event-stream; charset=utf-8", payload: anthropicSseResponse(model, result.answer) };
      return { status: 200, contentType: "application/json", payload: anthropicJsonResponse(model, result.answer) };
    }
  }
  return {
    status: 502,
    contentType: "application/json",
    payload: JSON.stringify({ error: { message: "no free provider could serve this request", type: "api_error", code: "no_provider" } }),
  };
}

function modelsResponse(state: ServerState, req: IncomingMessage): { status: number; contentType: string; payload: string } {
  const ids = freeModelIds(state.catalog);
  const anthropic = req.headers["x-api-key"] !== undefined && req.headers.authorization === undefined;
  if (anthropic) {
    return {
      status: 200,
      contentType: "application/json",
      payload: JSON.stringify({
        data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z", free: true })),
        has_more: false,
      }),
    };
  }
  return {
    status: 200,
    contentType: "application/json",
    payload: JSON.stringify({
      object: "list",
      data: ids.map((id) => ({ id, object: "model", created: 0, owned_by: "aggregator", free: true })),
    }),
  };
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function send(res: ServerResponse, status: number, contentType: string, payload: string): void {
  res.writeHead(status, { "content-type": contentType, "content-length": Buffer.byteLength(payload) });
  res.end(payload);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollWork(envUrl: string): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, "")}/work`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const body: unknown = await res.json();
        if (isRecord(body) && body.done === true) return;
      }
    } catch {
      // The env may not be up yet; keep polling.
    }
    await sleep(100);
  }
}

async function main(): Promise<void> {
  const listOnly = process.argv.includes("--list");
  const envUrl = env("MAKEFUN_ENV_URL");
  const brief = await loadEffectiveBrief(readBrief(), envUrl);
  const catalog = await loadCatalog(brief);
  if (listOnly) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (envUrl === undefined) {
    process.stderr.write("MAKEFUN_ENV_URL is not set; nothing to serve\n");
    process.exitCode = 1;
    return;
  }
  const state: ServerState = { brief, catalog, router: new RouterState(), served: 0 };
  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://server");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          const answer = modelsResponse(state, req);
          send(res, answer.status, answer.contentType, answer.payload);
          return;
        }
        if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
          const raw = await readBody(req);
          let body: unknown;
          try {
            body = JSON.parse(raw);
          } catch {
            send(res, 400, "application/json", JSON.stringify({ error: { message: "the request body is not one JSON object", type: "invalid_request_error", code: "invalid_request" } }));
            return;
          }
          if (!isRecord(body)) {
            send(res, 400, "application/json", JSON.stringify({ error: { message: "the request body is not one JSON object", type: "invalid_request_error", code: "invalid_request" } }));
            return;
          }
          const api = url.pathname === "/v1/chat/completions" ? "openai" : "anthropic";
          const answer = await handleRequest(state, api, body);
          send(res, answer.status, answer.contentType, answer.payload);
          return;
        }
        send(res, 404, "application/json", JSON.stringify({ error: { message: `no route ${req.method ?? ""} ${url.pathname}`, type: "invalid_request_error", code: "not_found" } }));
      } catch {
        try {
          send(res, 500, "application/json", JSON.stringify({ error: { message: "internal error", type: "api_error", code: "internal" } }));
        } catch {
          // The socket is already gone; nothing more to do.
        }
      }
    })();
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        reject(new Error("the server has no TCP address"));
        return;
      }
      resolve(address.port);
    });
  });
  const register = await fetch(`${envUrl.replace(/\/$/, "")}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(10000),
  });
  if (!register.ok) {
    process.stderr.write(`register answered ${register.status}\n`);
    process.exitCode = 1;
    server.close();
    return;
  }
  await pollWork(envUrl);
  server.close();
}

main().catch((err: unknown) => {
  process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
