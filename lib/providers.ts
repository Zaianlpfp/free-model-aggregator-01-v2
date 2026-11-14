// providers.ts — brief parsing, catalog probing and candidate ranking.
//
// The brief (MAKEFUN_PROMPT_FILE) carries three JSON documents: the provider
// directory, the sealed quality table and the workload manifest. The env's
// /instance route carries the same directory with absolute base URLs. The
// catalog we register is built by probing every provider's own catalog route
// and preserving each provider's listing order.

export type Wire = "openai" | "anthropic" | "gemini";
export type AuthScheme = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

export interface ModelRow {
  id: string;
  free: boolean;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  rating_milli: number;
}

export interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  key_env: string;
  key: string;
  base_url: string;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestEntry {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requires;
}

export interface Brief {
  directory: ProviderDir[];
  ratings: (ModelRow & { provider: string })[];
  manifest: ManifestEntry[];
}

export interface Candidate {
  provider: string;
  model: string;
  rating_milli: number;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asString(v: unknown, fallback: string): string {
  return typeof v === "string" ? v : fallback;
}

function asBool(v: unknown): boolean {
  return v === true;
}

function asIntOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Parse the brief text into its three JSON documents. */
export function parseBrief(text: string): Brief {
  const brief: Brief = { directory: [], ratings: [], manifest: [] };
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{"schema":"free-model-aggregator-01/')) continue;
    let doc: unknown;
    try {
      doc = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (!isRecord(doc) || typeof doc.schema !== "string") continue;
    if (doc.schema === "free-model-aggregator-01/providers/v1" && Array.isArray(doc.providers)) {
      for (const raw of doc.providers) {
        if (!isRecord(raw)) continue;
        brief.directory.push({
          id: asString(raw.id, ""),
          name: asString(raw.name, ""),
          wire: (raw.wire as Wire) ?? "openai",
          auth: (raw.auth as AuthScheme) ?? "none",
          key_env: asString(raw.key_env, ""),
          key: asString(raw.key, ""),
          base_url: asString(raw.path, ""),
        });
      }
    } else if (doc.schema === "free-model-aggregator-01/ratings/v1" && Array.isArray(doc.ratings)) {
      for (const raw of doc.ratings) {
        if (!isRecord(raw)) continue;
        brief.ratings.push({
          id: asString(raw.model, ""),
          provider: asString(raw.provider, ""),
          free: asBool(raw.free),
          context_length: asIntOrNull(raw.context_length),
          tools: asBool(raw.tools),
          vision: asBool(raw.vision),
          json: asBool(raw.json),
          rating_milli: typeof raw.rating_milli === "number" ? raw.rating_milli : 0,
        });
      }
    } else if (doc.schema === "free-model-aggregator-01/workload/v1" && Array.isArray(doc.manifest)) {
      for (const raw of doc.manifest) {
        if (!isRecord(raw)) continue;
        const req: Requires = {};
        if (isRecord(raw.requires)) {
          if (raw.requires.tools === true) req.tools = true;
          if (raw.requires.vision === true) req.vision = true;
          if (raw.requires.json === true) req.json = true;
          if (typeof raw.requires.min_context === "number") req.min_context = raw.requires.min_context;
        }
        brief.manifest.push({
          id: asString(raw.id, ""),
          api: raw.api === "anthropic" ? "anthropic" : "openai",
          stream: raw.stream === true,
          weight: typeof raw.weight === "number" ? raw.weight : 1,
          requires: req,
        });
      }
    }
  }
  return brief;
}

/** Auth headers for one provider, from its own scheme and key. */
export function authHeaders(p: ProviderDir): Record<string, string> {
  if (p.auth === "bearer") return { authorization: `Bearer ${p.key}` };
  if (p.auth === "x-api-key") return { "x-api-key": p.key, "anthropic-version": "2023-06-01" };
  if (p.auth === "x-goog-api-key") return { "x-goog-api-key": p.key };
  return {};
}

/** The catalog route path for a provider's wire. */
function catalogPath(p: ProviderDir): string {
  if (p.wire === "gemini") return "/v1beta/models";
  return "/v1/models";
}

/** Probe one provider's catalog route and return its models in listed order. */
export async function probeProvider(p: ProviderDir): Promise<CatalogModel[]> {
  const url = p.base_url.replace(/\/$/, "") + catalogPath(p);
  const res = await fetch(url, { headers: authHeaders(p), signal: AbortSignal.timeout(5000) });
  if (!res.ok) throw new Error(`catalog ${p.id} answered ${res.status}`);
  const body: unknown = await res.json();
  if (!isRecord(body)) throw new Error(`catalog ${p.id} is not an object`);
  const out: CatalogModel[] = [];
  if (p.wire === "gemini") {
    if (!Array.isArray(body.models)) throw new Error(`catalog ${p.id} holds no models`);
    for (const raw of body.models) {
      if (!isRecord(raw) || typeof raw.name !== "string") continue;
      const id = raw.name.startsWith("models/") ? raw.name.slice("models/".length) : raw.name;
      out.push({ id, free: asBool(raw.free), context_length: asIntOrNull(raw.context_length) });
    }
    return out;
  }
  if (!Array.isArray(body.data)) throw new Error(`catalog ${p.id} holds no data`);
  for (const raw of body.data) {
    if (!isRecord(raw) || typeof raw.id !== "string") continue;
    out.push({ id: raw.id, free: asBool(raw.free), context_length: asIntOrNull(raw.context_length) });
  }
  return out;
}

/**
 * Build the catalog document by probing every provider's catalog route.
 * Falls back to the brief's ratings table for a provider whose route fails,
 * so one failed probe never loses the whole catalog.
 */
export async function buildCatalog(brief: Brief): Promise<Catalog> {
  const providers: CatalogProvider[] = [];
  for (const p of brief.directory) {
    let models: CatalogModel[];
    try {
      models = await probeProvider(p);
    } catch {
      models = brief.ratings
        .filter((r) => r.provider === p.id)
        .map((r) => ({ id: r.id, free: r.free, context_length: r.context_length }));
    }
    providers.push({ id: p.id, base_url: p.base_url, key_env: p.key_env, wire: p.wire, models });
  }
  return { providers };
}

/** Catalog built purely from the brief, with no probing (for --list fallback). */
export function catalogFromBrief(brief: Brief): Catalog {
  return {
    providers: brief.directory.map((p) => ({
      id: p.id,
      base_url: p.base_url,
      key_env: p.key_env,
      wire: p.wire,
      models: brief.ratings
        .filter((r) => r.provider === p.id)
        .map((r) => ({ id: r.id, free: r.free, context_length: r.context_length })),
    })),
  };
}

/** The wire of a provider id, from the directory. */
export function wireOf(brief: Brief, providerId: string): Wire {
  return brief.directory.find((p) => p.id === providerId)?.wire ?? "openai";
}

/** The directory entry of a provider id. */
export function providerOf(brief: Brief, providerId: string): ProviderDir | undefined {
  return brief.directory.find((p) => p.id === providerId);
}

/**
 * Routing state: which models and providers have proven dead, and how many
 * calls each provider has taken. A model refused as unknown or paid is never
 * picked again; a provider that failed two different models is dead for the
 * rest of the trial; a transient refusal may be retried once on the same
 * model before it counts as dead.
 */
export class RouterState {
  deadModels = new Set<string>();
  deadProviders = new Set<string>();
  private failedModelsPerProvider = new Map<string, Set<string>>();

  key(provider: string, model: string): string {
    return `${provider}/${model}`;
  }

  isDead(provider: string, model: string): boolean {
    return this.deadProviders.has(provider) || this.deadModels.has(this.key(provider, model));
  }

  markModelDead(provider: string, model: string): void {
    this.deadModels.add(this.key(provider, model));
    let set = this.failedModelsPerProvider.get(provider);
    if (set === undefined) {
      set = new Set<string>();
      this.failedModelsPerProvider.set(provider, set);
    }
    set.add(model);
    if (set.size >= 2) this.deadProviders.add(provider);
  }

  markProviderDead(provider: string): void {
    this.deadProviders.add(provider);
  }
}

/**
 * Rank the free models that meet a request's requirements: highest sealed
 * rating first, ties broken deterministically by provider id, then model id.
 * A JSON-mode request never goes to an Anthropic-wire provider: its wire
 * cannot carry response_format, so the provider would echo json_mode false.
 * Paid models are never candidates.
 */
export function rankCandidates(
  brief: Brief,
  state: RouterState,
  requires: Requires,
): Candidate[] {
  const list: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (requires.tools === true && !row.tools) continue;
    if (requires.vision === true && !row.vision) continue;
    if (requires.json === true && !row.json) continue;
    if (requires.min_context !== undefined && (row.context_length ?? 0) < requires.min_context) continue;
    if (requires.json === true && wireOf(brief, row.provider) === "anthropic") continue;
    if (state.isDead(row.provider, row.id)) continue;
    list.push({ provider: row.provider, model: row.id, rating_milli: row.rating_milli });
  }
  list.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return list;
}

/** All free model ids across the catalog, deduplicated, first-seen order. */
export function freeModelIds(catalog: Catalog): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (m.free && !seen.has(m.id)) {
        seen.add(m.id);
        out.push(m.id);
      }
    }
  }
  return out;
}
