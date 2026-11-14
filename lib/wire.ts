// lib/wire.ts — the client wires and the three provider wires.
//
// The trial's client calls this server as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), JSON or server-sent events. Each request is
// translated into the routed provider's own wire (openai, anthropic or gemini), and the
// provider's answer is normalised to one shape — {text, stopFamily, inputTokens,
// outputTokens} — then rendered back in the wire the client used.
//
// The translations preserve every field the provider echoes back (system text, user
// text, tool names, max_tokens, stop sequences, image bytes, JSON mode), because the
// client compares that echo with what it sent: a field rewritten or dropped loses the
// request. Two subtleties the reference build taught me:
//   - an Anthropic-wire provider cannot carry JSON mode (its echo always says json_mode
//     false), so main.ts never routes a JSON-mode request to one;
//   - tool messages merge into the previous user message on the OpenAI and Gemini wires,
//     because those wires' echo reads the last user message, not a tool message; and a
//     user message of tool_result blocks alone keeps its own empty text on the Gemini
//     wire, because the Anthropic echo reads that message's text, which is empty there.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so
// the same body and the same provider answer give the same bytes on a replay. The two
// model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model this server serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
export function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`:
 * the system prompt becomes a system message, image blocks become image_url data URLs,
 * tool_use blocks become tool_calls, tool_result blocks become tool messages, tools and
 * tool_choice are rewritten, and the output cap, temperature, top_p, stop sequences and
 * streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    let sawToolResult = false;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        sawToolResult = true;
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    // A user message of tool_result blocks alone still counts as the last user message on
    // the client's side (its text is empty), so keep an empty user message after the tool
    // messages: without it the provider would echo an earlier user message's text instead.
    else if (sawToolResult) messages.push({ role: 'user', content: '' });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A data URL as its media type and base64 payload, or null when it is not one. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  return match === null ? null : { media: match[1] ?? 'image/png', data: match[2] ?? '' };
}

/** An OpenAI content part as an Anthropic content block (text or image); null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const data = dataUrlOf(part.image_url.url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
    return { type: 'image', source: { type: 'url', url: part.image_url.url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`.
 * Tool messages merge into the previous user message as tool_result blocks: the
 * provider's echo reads the last user message's text, and a tool message is not one,
 * so merging keeps the echo the client sent.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const lastUser = (): Json | null => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i]?.role === 'user') return messages[i] as Json;
    }
    return null;
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      const block: Json = { type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) };
      const user = lastUser();
      if (user !== null) {
        const content = Array.isArray(user.content) ? user.content : [];
        content.push(block);
        user.content = content;
      } else {
        messages.push({ role: 'user', content: [block] });
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
      }
      if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message: text and image parts become blocks in order.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks = raw.content.flatMap((p) => {
        if (!isRecord(p)) return [];
        const block = anthropicBlockOf(p);
        return block === null ? [] : [block];
      });
      messages.push({ role: 'user', content: blocks.length > 0 ? blocks : '' });
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** One Gemini content part from an OpenAI text or image part; null for the rest. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const data = dataUrlOf(part.image_url.url);
    if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
    return { fileData: { fileUri: part.image_url.url } };
  }
  return null;
}

/** The generationConfig of an OpenAI request: caps, sampling, stop sequences, JSON mode. */
function geminiConfigOfOpenAI(body: Json): Json {
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) config.responseMimeType = 'application/json';
  return config;
}

/** OpenAI tool definitions as Gemini function declarations. */
function geminiDeclarationsOfOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. Tool messages
 * merge into the previous user content as functionResponse parts: the provider's echo
 * reads the last user content's text, and a tool message is not a user message, so
 * merging keeps the echo the client sent.
 */
export function openAIToGemini(body: Json, _model: string): Json {
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const lastUserContent = (): Json | null => {
    for (let i = contents.length - 1; i >= 0; i--) {
      if (contents[i]?.role === 'user') return contents[i] as Json;
    }
    return null;
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    if (raw.role === 'tool') {
      const part: Json = { functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } };
      const user = lastUserContent();
      if (user !== null) {
        const parts = Array.isArray(user.parts) ? user.parts : [];
        parts.push(part);
        user.parts = parts;
      } else {
        contents.push({ role: 'user', parts: [part] });
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length === 0) parts.push({ text: '' });
      contents.push({ role: 'model', parts });
      continue;
    }
    if (typeof raw.content === 'string') {
      contents.push({ role: 'user', parts: [{ text: raw.content }] });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const parts = raw.content.flatMap((p) => {
        if (!isRecord(p)) return [];
        const part = geminiPartOfOpenAI(p);
        return part === null ? [] : [part];
      });
      contents.push({ role: 'user', parts: parts.length > 0 ? parts : [{ text: '' }] });
    }
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const declarations = geminiDeclarationsOfOpenAI(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfigOfOpenAI(body);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. A user message holding
 * only tool_result blocks becomes its own user content with no text: the provider's echo
 * reads the last user content's text, and the client's echo reads the last user message's
 * text, which is empty there, so a separate empty content keeps both empty.
 */
export function anthropicToGemini(body: Json, _model: string): Json {
  const systemParts: Json[] = [];
  const systemText = textOf(body.system);
  if (systemText !== '') systemParts.push({ text: systemText });
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length === 0) parts.push({ text: '' });
      contents.push({ role: 'model', parts });
      continue;
    }
    // A user message: text and image parts in one content, tool_result blocks beside them,
    // unless the message holds tool_result blocks alone, which get their own textless content.
    const parts: Json[] = [];
    const responses: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
          else if (src.type === 'url' && typeof src.url === 'string') parts.push({ fileData: { fileUri: src.url } });
        } else if (block.type === 'tool_result') {
          responses.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { result: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length === 0 && responses.length > 0) {
      contents.push({ role: 'user', parts: responses });
    } else {
      contents.push({ role: 'user', parts: [...parts, ...responses] });
    }
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The stop-reason family of one provider's stop value: 0 stop, 1 length, 2 tool calls, 3 refusal. */
export function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

/** One stop family as an OpenAI finish_reason. */
export function openAIStopOf(family: number): string {
  return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
}

/** One stop family as an Anthropic stop_reason. */
export function anthropicStopOf(family: number): string {
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

/** One provider answer, normalised: its text whole, its stop family and its usage counts. */
export interface UpstreamAnswer {
  text: string;
  stopFamily: number;
  inputTokens: number;
  outputTokens: number;
}

/** A Chat Completions JSON answer as an UpstreamAnswer. */
export function parseOpenAIJson(body: Json): UpstreamAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    stopFamily: stopFamilyOf(choice.finish_reason),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
  };
}

/** One SSE frame: its event name (null when unnamed) and its joined data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

export function sseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOf(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A Chat Completions SSE stream as an UpstreamAnswer. */
export function parseOpenAIStream(text: string): UpstreamAnswer {
  let out = '';
  let stopFamily = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = jsonOf(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stopFamily = stopFamilyOf(choice.finish_reason);
  }
  return { text: out, stopFamily, inputTokens, outputTokens };
}

/** An Anthropic Messages JSON answer as an UpstreamAnswer. */
export function parseAnthropicJson(body: Json): UpstreamAnswer | null {
  if (!Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    stopFamily: stopFamilyOf(body.stop_reason),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
  };
}

/** An Anthropic Messages SSE stream as an UpstreamAnswer. */
export function parseAnthropicStream(text: string): UpstreamAnswer {
  let out = '';
  let stopFamily = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(text)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopFamily = stopFamilyOf(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, stopFamily, inputTokens, outputTokens };
}

function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  const content = candidate !== null && isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

function geminiUsageOf(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent JSON answer as an UpstreamAnswer. */
export function parseGeminiJson(body: Json): UpstreamAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  if (candidates.length === 0) return null;
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsageOf(body);
  return { text: geminiTextOf(body), stopFamily: stopFamilyOf(candidate.finishReason), inputTokens: usage.input, outputTokens: usage.output };
}

/** A Gemini streamGenerateContent SSE stream as an UpstreamAnswer. */
export function parseGeminiStream(text: string): UpstreamAnswer {
  let out = '';
  let stopFamily = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(text)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    out += geminiTextOf(data);
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null && typeof candidate.finishReason === 'string') stopFamily = stopFamilyOf(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      const usage = geminiUsageOf(data);
      inputTokens = usage.input;
      outputTokens = usage.output;
    }
  }
  return { text: out, stopFamily, inputTokens, outputTokens };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An UpstreamAnswer as a Chat Completions JSON object. */
export function openAIJsonAnswer(model: string, ans: UpstreamAnswer): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: ans.text }, finish_reason: openAIStopOf(ans.stopFamily), logprobs: null }],
    usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
  };
}

/** An UpstreamAnswer as a Chat Completions SSE stream (text). */
export function openAIStreamAnswer(model: string, ans: UpstreamAnswer): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [
    `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`,
    `data: ${JSON.stringify(chunk({ content: ans.text }, null))}\n\n`,
    `data: ${JSON.stringify(chunk({}, openAIStopOf(ans.stopFamily)))}\n\n`,
    `data: ${JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens } })}\n\n`,
    'data: [DONE]\n\n',
  ];
  return frames.join('');
}

/** An UpstreamAnswer as an Anthropic Messages JSON object. */
export function anthropicJsonAnswer(model: string, ans: UpstreamAnswer): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: ans.text }],
    stop_reason: anthropicStopOf(ans.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

/** An UpstreamAnswer as an Anthropic Messages SSE stream (text). */
export function anthropicStreamAnswer(model: string, ans: UpstreamAnswer): string {
  return [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: ans.inputTokens, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: ans.text } }),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStopOf(ans.stopFamily), stop_sequence: null }, usage: { output_tokens: ans.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ].join('');
}
