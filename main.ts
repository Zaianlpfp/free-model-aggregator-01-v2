// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief, probes every provider's catalog route, builds one
// catalog, registers it with the env, then answers the client's 20 requests: each with
// the best-rated free model that meets its requirements, failing over to the next-best
// free model when a provider refuses, and exiting 0 when the env says the workload is
// done.
//
// Routing: every free candidate that meets the request's requirements is ranked by the
// quality table's rating_milli (best first, ties by provider id then model id, so the
// choice is a pure function of the brief and the answers seen so far). A provider that
// answers "budget is spent" or "is out" is dead for the rest of the trial; a provider
// that "refused this call" stays eligible — its failure schedule may have passed by the
// next request. Calls here are strictly sequential, one at a time, so no
// concurrent_requests refusal can occur.
//
// Wires: every upstream answer is normalised to {text, stopFamily, inputTokens,
// outputTokens} and then rendered in the wire the client used, JSON or server-sent
// events. `node main.ts --list` prints the catalog document and exits 0, byte for byte
// as registered.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicStreamAnswer,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIJsonAnswer,
  openAIRequestToAnthropic,
  openAIStreamAnswer,
  openAIToGemini,
  openAIModelList,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Providers a refusal proved are out of budget or in outage: never dialled again this trial. */
const deadProviders = new Set<string>();
/** Models a provider refused as unknown or paid: never picked again this trial. */
const badModels = new Set<string>();
/** How many client requests were served, and how many provider calls they cost: logged at exit. */
let servedRequests = 0;
let upstreamCalls = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in probe order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first. Ties break by provider
 * id then model id, so the order is a pure function of the brief. A JSON-mode request
 * never goes to an Anthropic-wire provider: its wire cannot carry response_format, so
 * its echo would say the JSON mode the client asked for was absent.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (deadProviders.has(probe.provider.id)) continue;
    if (requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      if (badModels.has(`${probe.provider.id}/${model.id}`)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || byId(a.model, b.model));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The error message of a provider's error body, in any of the three wires' envelopes. */
function providerErrorMessage(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
    if (isRecord(value) && typeof value.message === 'string') return value.message;
  } catch {
    // Not JSON: the raw text is the message.
  }
  return text.slice(0, 500);
}

interface ProviderResult {
  ok: boolean;
  answer: UpstreamAnswer | null;
  /** True when the provider is out of budget or in outage: never dial it again this trial. */
  dead: boolean;
  /** True when the model itself was refused as unknown or paid: never pick it again. */
  badModel: boolean;
  detail: string;
}

/**
 * One inference call to one provider's model, in that provider's wire, normalised to an
 * UpstreamAnswer. `clientApi` is the wire the client used; `body` is the client's request body.
 */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<ProviderResult> {
  const provider = candidate.probe.provider;
  const model = candidate.model.id;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'anthropic' ? anthropicToOpenAI(body, model) : { ...body, model };
    if (stream) upstreamBody.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model } : openAIRequestToAnthropic(body, model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    upstreamBody = clientApi === 'anthropic' ? anthropicToGemini(body, model) : openAIToGemini(body, model);
  }
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
    upstreamCalls += 1;
  } catch (err) {
    return { ok: false, answer: null, dead: false, badModel: false, detail: `the provider could not be reached: ${(err as Error).message}` };
  }
  const text = await upstream.text();
  if (!upstream.ok) {
    const message = providerErrorMessage(text);
    const dead = /budget is spent/i.test(message) || /\bis out\b/i.test(message);
    const badModel = upstream.status === 400 && /paid model|no model|unknown model/i.test(message);
    return { ok: false, answer: null, dead, badModel, detail: `HTTP ${upstream.status}: ${message}` };
  }
  let answer: UpstreamAnswer | null = null;
  try {
    if (provider.wire === 'openai') {
      answer = stream ? parseOpenAIStream(text) : parseOpenAIJson(JSON.parse(text));
    } else if (provider.wire === 'anthropic') {
      answer = stream ? parseAnthropicStream(text) : parseAnthropicJson(JSON.parse(text));
    } else {
      answer = stream ? parseGeminiStream(text) : parseGeminiJson(JSON.parse(text));
    }
  } catch {
    answer = null;
  }
  if (answer === null) return { ok: false, answer: null, dead: false, badModel: false, detail: 'the provider answered a body this server could not read' };
  return { ok: true, answer, dead: false, badModel: false, detail: '' };
}

/**
 * Serve one client request: try the ranked free candidates in order until one answers,
 * then render its answer in the client's wire. A provider that refuses transiently
 * ("refused this call": its failure schedule, not its budget) is tried once more on the
 * same model: the next call number usually serves, and keeping the best-rated model is
 * worth one extra call. A provider that is out of budget or in outage, or a model
 * refused as unknown or paid, is never retried. Returns null when no candidate answered.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<{ model: string; answer: UpstreamAnswer } | null> {
  const candidates = candidatesFor(requirementsFor(body));
  for (const candidate of candidates) {
    for (let attempt = 0; attempt < 2; attempt++) {
      const result = await callProvider(candidate, clientApi, body, stream);
      if (result.ok && result.answer !== null) {
        servedRequests += 1;
        return { model: candidate.model.id, answer: result.answer };
      }
      if (result.dead) {
        deadProviders.add(candidate.probe.provider.id);
        log(`${candidate.probe.provider.id} is out: ${result.detail}`);
        break;
      }
      if (result.badModel) {
        badModels.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
        log(`${candidate.probe.provider.id}/${candidate.model.id} refused as a model: ${result.detail}`);
        break;
      }
      log(`${candidate.probe.provider.id}/${candidate.model.id} refused (attempt ${attempt + 1}): ${result.detail}`);
    }
  }
  return null;
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('openai', body, stream);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  if (stream) return sendSse(res, openAIStreamAnswer(served.model, served.answer));
  sendJson(res, 200, openAIJsonAnswer(served.model, served.answer));
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('anthropic', body, stream);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  if (stream) return sendSse(res, anthropicStreamAnswer(served.model, served.answer));
  sendJson(res, 200, anthropicJsonAnswer(served.model, served.answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          log(`done: served ${servedRequests} requests with ${upstreamCalls} provider calls`);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
