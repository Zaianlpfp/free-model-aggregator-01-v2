// wire.ts — translation between the client's wire and each provider's wire,
// the provider calls themselves, and the answers rebuilt in the client's wire.
//
// The simulated providers answer every request with one fixed text: an
// "ANSWER ECHO" carrying the sha8 of the system text, the user text, the
// tool names, max_tokens, the stop sequences, the image sha and the JSON
// flag, exactly as the provider parsed them. So every field the client sent
// must arrive at the provider byte for byte, in whichever wire it speaks:
// system text, user text, tool names in order, max_tokens, stops, images and
// JSON mode. The provider's usage counts are passed back through unchanged,
// and its stop reason is mapped to the client's wire (stop / end_turn).
//
// Upstream calls are always made non-streaming: a streamed client request is
// answered by re-emitting the provider's whole answer as SSE. The client
// sends no stream_options.include_usage, so a streamed upstream call would
// carry no usage chunk and the usage check would fail; calling the provider
// non-streaming returns its usage in JSON, which the SSE re-emit then
// carries in its final chunk.

import type { ProviderDir } from "./providers.ts";
import { authHeaders } from "./providers.ts";

export interface NormImage {
  kind: "base64" | "url";
  mediaType: string;
  data: string;
}

export interface NormToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface NormMessage {
  role: "system" | "user" | "assistant" | "tool";
  texts: string[];
  images: NormImage[];
  toolCalls: NormToolCall[];
  toolCallId: string;
  toolName: string;
}

export interface NormTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface NormRequest {
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: "auto" | "none" | { name: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  stops: string[];
  jsonMode: boolean;
}

export interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
  stop: string;
}

export interface CallFailure {
  status: number;
  code: string;
  message: string;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asString(v: unknown, fallback: string): string {
  return typeof v === "string" ? v : fallback;
}

function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

/** The text of a block list, text blocks joined with a newline. */
function blocksText(content: unknown): string[] {
  if (typeof content === "string") return content === "" ? [] : [content];
  if (!Array.isArray(content)) return [];
  const out: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") out.push(block.text);
  }
  return out;
}

function joinTexts(texts: string[]): string {
  return texts.join("\n");
}

function emptyMessage(role: NormMessage["role"]): NormMessage {
  return { role, texts: [], images: [], toolCalls: [], toolCallId: "", toolName: "" };
}

/** Split a data URL into its media type and base64 payload. */
function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1] === undefined || m[1] === "" ? "application/octet-stream" : m[1], data: m[2] };
}

/** Extract the normalized form of an OpenAI Chat Completions body. */
export function normalizeOpenAI(body: Record<string, unknown>): NormRequest {
  const messages: NormMessage[] = [];
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      const role = raw.role;
      if (role === "system" || role === "user" || role === "assistant" || role === "tool") {
        const msg = emptyMessage(role);
        msg.texts = blocksText(raw.content);
        if (role === "user" && Array.isArray(raw.content)) {
          for (const block of raw.content) {
            if (!isRecord(block) || block.type !== "image_url" || !isRecord(block.image_url)) continue;
            const url = asString(block.image_url.url, "");
            if (url === "") continue;
            const data = parseDataUrl(url);
            if (data !== null) msg.images.push({ kind: "base64", mediaType: data.mediaType, data: data.data });
            else msg.images.push({ kind: "url", mediaType: "", data: url });
          }
        }
        if (role === "assistant" && Array.isArray(raw.tool_calls)) {
          for (const call of raw.tool_calls) {
            if (!isRecord(call) || !isRecord(call.function)) continue;
            let args: Record<string, unknown> = {};
            if (typeof call.function.arguments === "string") {
              try {
                const parsed: unknown = JSON.parse(call.function.arguments);
                if (isRecord(parsed)) args = parsed;
              } catch {
                args = {};
              }
            } else if (isRecord(call.function.arguments)) {
              args = call.function.arguments;
            }
            msg.toolCalls.push({ id: asString(call.id, ""), name: asString(call.function.name, ""), args });
          }
        }
        if (role === "tool") {
          msg.toolCallId = asString(raw.tool_call_id, "");
          msg.toolName = "";
        }
        messages.push(msg);
      }
    }
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function)) continue;
      tools.push({
        name: asString(raw.function.name, ""),
        description: asString(raw.function.description, ""),
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : {},
      });
    }
  }
  let toolChoice: NormRequest["toolChoice"] = null;
  if (body.tool_choice === "auto" || body.tool_choice === "none") toolChoice = body.tool_choice;
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function)) {
    toolChoice = { name: asString(body.tool_choice.function.name, "") };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    messages,
    tools,
    toolChoice,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    temperature: asNumber(body.temperature),
    stops: stringList(body.stop),
    jsonMode: format === "json_object" || format === "json_schema",
  };
}

/** Extract the normalized form of an Anthropic Messages body. */
export function normalizeAnthropic(body: Record<string, unknown>): NormRequest {
  const messages: NormMessage[] = [];
  const systemTexts = blocksText(body.system);
  if (systemTexts.length > 0) {
    const sys = emptyMessage("system");
    sys.texts = systemTexts;
    messages.push(sys);
  }
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      const role = raw.role;
      if (role !== "user" && role !== "assistant") continue;
      const msg = emptyMessage(role);
      msg.texts = blocksText(raw.content);
      // Tool results travel inside a user message in the Anthropic wire;
      // they become their own normalized message after the user message.
      const toolMessages: NormMessage[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === "image" && isRecord(block.source)) {
            if (block.source.type === "base64" && typeof block.source.data === "string") {
              msg.images.push({ kind: "base64", mediaType: asString(block.source.media_type, "image/png"), data: block.source.data });
            } else if (block.source.type === "url" && typeof block.source.url === "string") {
              msg.images.push({ kind: "url", mediaType: "", data: block.source.url });
            }
          } else if (block.type === "tool_use") {
            const input = isRecord(block.input) ? block.input : {};
            msg.toolCalls.push({ id: asString(block.id, ""), name: asString(block.name, ""), args: input });
          } else if (block.type === "tool_result") {
            const tool = emptyMessage("tool");
            tool.toolCallId = asString(block.tool_use_id, "");
            tool.toolName = "";
            tool.texts = typeof block.content === "string" ? [block.content] : blocksText(block.content);
            toolMessages.push(tool);
          }
        }
      }
      messages.push(msg);
      for (const tool of toolMessages) messages.push(tool);
    }
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      tools.push({
        name: asString(raw.name, ""),
        description: asString(raw.description, ""),
        parameters: isRecord(raw.input_schema) ? raw.input_schema : {},
      });
    }
  }
  let toolChoice: NormRequest["toolChoice"] = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === "auto") toolChoice = "auto";
    else if (body.tool_choice.type === "none") toolChoice = "none";
    else if (body.tool_choice.type === "tool" && typeof body.tool_choice.name === "string") {
      toolChoice = { name: body.tool_choice.name };
    }
  }
  return {
    messages,
    tools,
    toolChoice,
    maxTokens: asNumber(body.max_tokens),
    temperature: asNumber(body.temperature),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
  };
}

/** The system text: every system message's text, joined with a newline. */
export function systemTextOf(norm: NormRequest): string {
  const parts: string[] = [];
  for (const m of norm.messages) if (m.role === "system") parts.push(joinTexts(m.texts));
  return parts.join("\n");
}

/** Resolve a tool message's function name from the assistant calls before it. */
function toolNameFor(norm: NormRequest, index: number, callId: string): string {
  for (let i = index - 1; i >= 0; i--) {
    for (const call of norm.messages[i].toolCalls) if (call.id === callId) return call.name;
  }
  return callId;
}

// ---------------------------------------------------------------------------
// Building provider bodies
// ---------------------------------------------------------------------------

function openAIMessages(norm: NormRequest): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (const m of norm.messages) {
    if (m.role === "system") {
      const text = joinTexts(m.texts);
      if (text !== "") out.push({ role: "system", content: text });
    } else if (m.role === "user") {
      const text = joinTexts(m.texts);
      if (m.images.length === 0) {
        out.push({ role: "user", content: text });
      } else {
        const content: Record<string, unknown>[] = [];
        if (text !== "") content.push({ type: "text", text });
        for (const img of m.images) {
          const url = img.kind === "base64" ? `data:${img.mediaType};base64,${img.data}` : img.data;
          content.push({ type: "image_url", image_url: { url } });
        }
        out.push({ role: "user", content });
      }
    } else if (m.role === "assistant") {
      const text = joinTexts(m.texts);
      const msg: Record<string, unknown> = { role: "assistant", content: m.toolCalls.length > 0 && text === "" ? null : text };
      if (m.toolCalls.length > 0) {
        msg.tool_calls = m.toolCalls.map((c) => ({
          id: c.id,
          type: "function",
          function: { name: c.name, arguments: JSON.stringify(c.args) },
        }));
      }
      out.push(msg);
    } else if (m.role === "tool") {
      out.push({ role: "tool", tool_call_id: m.toolCallId, content: joinTexts(m.texts) });
    }
  }
  return out;
}

function openAITools(norm: NormRequest): Record<string, unknown>[] {
  return norm.tools.map((t) => ({
    type: "function",
    function: { name: t.name, description: t.description, parameters: t.parameters },
  }));
}

function openAIToolChoice(norm: NormRequest): unknown {
  if (norm.toolChoice === "auto") return "auto";
  if (norm.toolChoice === "none") return "none";
  if (norm.toolChoice !== null) return { type: "function", function: { name: norm.toolChoice.name } };
  return undefined;
}

/** Build an OpenAI-wire provider body from the normalized request. */
export function toOpenAI(norm: NormRequest, model: string): Record<string, unknown> {
  const body: Record<string, unknown> = { model, messages: openAIMessages(norm), stream: false };
  if (norm.tools.length > 0) body.tools = openAITools(norm);
  const choice = openAIToolChoice(norm);
  if (choice !== undefined) body.tool_choice = choice;
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.stops.length > 0) body.stop = norm.stops.length === 1 ? norm.stops[0] : norm.stops;
  if (norm.jsonMode) body.response_format = { type: "json_object" };
  return body;
}

function anthropicContent(m: NormMessage): Record<string, unknown>[] {
  const content: Record<string, unknown>[] = [];
  const text = joinTexts(m.texts);
  if (text !== "") content.push({ type: "text", text });
  for (const img of m.images) {
    if (img.kind === "base64") {
      content.push({ type: "image", source: { type: "base64", media_type: img.mediaType, data: img.data } });
    } else {
      content.push({ type: "image", source: { type: "url", url: img.data } });
    }
  }
  for (const call of m.toolCalls) {
    content.push({ type: "tool_use", id: call.id, name: call.name, input: call.args });
  }
  return content;
}

/** Build an Anthropic-wire provider body from the normalized request. */
export function toAnthropic(norm: NormRequest, model: string): Record<string, unknown> {
  const system = systemTextOf(norm);
  const messages: Record<string, unknown>[] = [];
  let pendingTools: Record<string, unknown>[] = [];
  const flushTools = () => {
    if (pendingTools.length > 0) {
      messages.push({ role: "user", content: pendingTools });
      pendingTools = [];
    }
  };
  for (const m of norm.messages) {
    if (m.role === "system") continue;
    if (m.role === "tool") {
      pendingTools.push({ type: "tool_result", tool_use_id: m.toolCallId, content: joinTexts(m.texts) });
      continue;
    }
    const content = anthropicContent(m);
    if (pendingTools.length > 0 && m.role === "user") {
      // Tool results belong in a user message: merge them ahead of the text.
      messages.push({ role: "user", content: [...pendingTools, ...content] });
      pendingTools = [];
    } else {
      flushTools();
      messages.push({ role: m.role, content });
    }
  }
  flushTools();
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (system !== "") body.system = system;
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (norm.toolChoice === "auto") body.tool_choice = { type: "auto" };
  else if (norm.toolChoice === "none") body.tool_choice = { type: "none" };
  else if (norm.toolChoice !== null) body.tool_choice = { type: "tool", name: norm.toolChoice.name };
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  return body;
}

function geminiParts(m: NormMessage): Record<string, unknown>[] {
  const parts: Record<string, unknown>[] = [];
  const text = joinTexts(m.texts);
  if (text !== "") parts.push({ text });
  for (const img of m.images) {
    if (img.kind === "base64") parts.push({ inlineData: { mimeType: img.mediaType, data: img.data } });
    else parts.push({ fileData: { fileUri: img.data } });
  }
  for (const call of m.toolCalls) {
    parts.push({ functionCall: { name: call.name, args: call.args } });
  }
  return parts;
}

/** Build a Gemini-wire provider body from the normalized request. */
export function toGemini(norm: NormRequest, model: string): { url: string; body: Record<string, unknown> } {
  const system = systemTextOf(norm);
  const contents: Record<string, unknown>[] = [];
  norm.messages.forEach((m, index) => {
    if (m.role === "system") return;
    if (m.role === "tool") {
      contents.push({
        role: "function",
        parts: [{ functionResponse: { name: toolNameFor(norm, index, m.toolCallId), response: { content: joinTexts(m.texts) } } }],
      });
      return;
    }
    contents.push({ role: m.role === "assistant" ? "model" : "user", parts: geminiParts(m) });
  });
  const body: Record<string, unknown> = { contents };
  if (system !== "") body.systemInstruction = { parts: [{ text: system }] };
  if (norm.tools.length > 0) {
    body.tools = [{ functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  if (norm.toolChoice === "auto") {
    body.toolConfig = { functionCallingConfig: { mode: "AUTO" } };
  } else if (norm.toolChoice === "none") {
    body.toolConfig = { functionCallingConfig: { mode: "NONE" } };
  } else if (norm.toolChoice !== null) {
    body.toolConfig = { functionCallingConfig: { mode: "ANY", allowedFunctionNames: [norm.toolChoice.name] } };
  }
  const config: Record<string, unknown> = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { url: model, body };
}

// ---------------------------------------------------------------------------
// Provider calls
// ---------------------------------------------------------------------------

/**
 * The refusal's code and message, in the provider wire's error envelope.
 * The message names the cause: "<Name>'s free-tier budget is spent",
 * "<Name> is out" (outage) or "<Name> refused this call" (failure schedule).
 * The code alone does not distinguish them.
 */
function errorInfoOf(wire: string, body: unknown): { code: string; message: string } {
  if (!isRecord(body) || !isRecord(body.error)) return { code: "", message: "" };
  const error = body.error;
  if (wire === "openai") return { code: asString(error.code, ""), message: asString(error.message, "") };
  if (wire === "anthropic") return { code: asString(error.type, ""), message: asString(error.message, "") };
  return { code: asString(error.status, ""), message: asString(error.message, "") };
}

function textFromOpenAIMessage(message: unknown): string {
  if (!isRecord(message)) return "";
  if (typeof message.content === "string") return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .filter(isRecord)
      .filter((b) => b.type === "text" && typeof b.text === "string")
      .map((b) => b.text as string)
      .join("");
  }
  return "";
}

function textFromAnthropicContent(content: unknown): string {
  if (!Array.isArray(content)) return "";
  return content
    .filter(isRecord)
    .filter((b) => b.type === "text" && typeof b.text === "string")
    .map((b) => b.text as string)
    .join("");
}

function textFromGeminiParts(parts: unknown): string {
  if (!Array.isArray(parts)) return "";
  return parts
    .filter(isRecord)
    .filter((p) => typeof p.text === "string")
    .map((p) => p.text as string)
    .join("");
}

/**
 * Call one provider in its own wire with the normalized request, always
 * non-streaming. Returns the provider's answer text, usage and stop reason,
 * or the refusal status and code.
 */
export async function callProvider(
  provider: ProviderDir,
  model: string,
  norm: NormRequest,
): Promise<{ ok: true; answer: ProviderAnswer } | { ok: false; failure: CallFailure }> {
  const base = provider.base_url.replace(/\/$/, "");
  const headers: Record<string, string> = { ...authHeaders(provider), "content-type": "application/json" };
  let url: string;
  let payload: Record<string, unknown>;
  if (provider.wire === "openai") {
    url = `${base}/v1/chat/completions`;
    payload = toOpenAI(norm, model);
  } else if (provider.wire === "anthropic") {
    url = `${base}/v1/messages`;
    payload = toAnthropic(norm, model);
  } else {
    const built = toGemini(norm, model);
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    payload = built.body;
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(25000),
    });
  } catch {
    return { ok: false, failure: { status: 0, code: "network_error", message: "" } };
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const info = errorInfoOf(provider.wire, body);
    return { ok: false, failure: { status: res.status, code: info.code, message: info.message } };
  }
  if (!isRecord(body)) return { ok: false, failure: { status: res.status, code: "bad_answer", message: "" } };
  if (provider.wire === "openai") {
    const choice = Array.isArray(body.choices) ? body.choices[0] : undefined;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      ok: true,
      answer: {
        text: textFromOpenAIMessage(isRecord(choice) ? choice.message : undefined),
        input: asNumber(usage.prompt_tokens) ?? 0,
        output: asNumber(usage.completion_tokens) ?? 0,
        stop: isRecord(choice) ? asString(choice.finish_reason, "stop") : "stop",
      },
    };
  }
  if (provider.wire === "anthropic") {
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      ok: true,
      answer: {
        text: textFromAnthropicContent(body.content),
        input: asNumber(usage.input_tokens) ?? 0,
        output: asNumber(usage.output_tokens) ?? 0,
        stop: asString(body.stop_reason, "end_turn"),
      },
    };
  }
  const candidate = Array.isArray(body.candidates) ? body.candidates[0] : undefined;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    ok: true,
    answer: {
      text: textFromGeminiParts(isRecord(candidate) && isRecord(candidate.content) ? candidate.content.parts : undefined),
      input: asNumber(usage.promptTokenCount) ?? 0,
      output: asNumber(usage.candidatesTokenCount) ?? 0,
      stop: isRecord(candidate) ? asString(candidate.finishReason, "STOP") : "STOP",
    },
  };
}

// ---------------------------------------------------------------------------
// Answers rebuilt in the client's wire
// ---------------------------------------------------------------------------

function openAIStop(stop: string): string {
  if (stop === "length" || stop === "max_tokens" || stop === "MAX_TOKENS") return "length";
  if (stop === "tool_calls" || stop === "tool_use" || stop === "function_call") return "tool_calls";
  if (stop === "content_filter" || stop === "refusal" || stop === "SAFETY") return "content_filter";
  return "stop";
}

function anthropicStop(stop: string): string {
  if (stop === "length" || stop === "max_tokens" || stop === "MAX_TOKENS") return "max_tokens";
  if (stop === "tool_calls" || stop === "tool_use" || stop === "function_call") return "tool_use";
  if (stop === "content_filter" || stop === "refusal" || stop === "SAFETY") return "refusal";
  return "end_turn";
}

function splitText(text: string): string[] {
  if (text === "") return [""];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 120) out.push(text.slice(at, at + 120));
  return out;
}

export function openAIJsonResponse(model: string, answer: ProviderAnswer): string {
  return JSON.stringify({
    id: "chatcmpl-agg",
    object: "chat.completion",
    created: 0,
    model,
    choices: [{ index: 0, message: { role: "assistant", content: answer.text }, finish_reason: openAIStop(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  });
}

export function openAISseResponse(model: string, answer: ProviderAnswer): string {
  const chunk = (delta: Record<string, unknown>, finish: string | null): string =>
    `data: ${JSON.stringify({ id: "chatcmpl-agg", object: "chat.completion.chunk", created: 0, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  const frames: string[] = [chunk({ role: "assistant", content: "" }, null)];
  for (const piece of splitText(answer.text)) frames.push(chunk({ content: piece }, null));
  frames.push(chunk({}, openAIStop(answer.stop)));
  frames.push(
    `data: ${JSON.stringify({ id: "chatcmpl-agg", object: "chat.completion.chunk", created: 0, model, choices: [], usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output } })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

export function anthropicJsonResponse(model: string, answer: ProviderAnswer): string {
  return JSON.stringify({
    id: "msg_agg",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  });
}

export function anthropicSseResponse(model: string, answer: ProviderAnswer): string {
  const frame = (event: string, data: Record<string, unknown>): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    frame("message_start", {
      type: "message_start",
      message: { id: "msg_agg", type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.input, output_tokens: 0 } },
    }),
    frame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(frame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(
    frame("content_block_stop", { type: "content_block_stop", index: 0 }),
    frame("message_delta", { type: "message_delta", delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null }, usage: { output_tokens: answer.output } }),
    frame("message_stop", { type: "message_stop" }),
  );
  return frames.join("");
}
