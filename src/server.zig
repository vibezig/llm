//! OpenAI-compatible HTTP API + a single-file web UI embedded in the binary.
//!
//! Routes:
//!   GET  /                     chat UI
//!   GET  /health               {"ok":true}
//!   GET  /v1/models            model list (OpenAI schema)
//!   GET  /v1/models/{id}       single model
//!   POST /v1/chat/completions  chat completion, `stream: true` for SSE
//!   POST /v1/completions       legacy text completion, same body shape
//!
//! Any OpenAI client works by pointing its base URL at this server:
//!
//!   export OPENAI_BASE_URL=http://localhost:8080/v1
//!   export OPENAI_API_KEY=local
//!
//! The API key is accepted and ignored: there is nothing to authenticate.
const std = @import("std");
const model_mod = @import("model.zig");
const gpt = @import("gpt.zig");
const tokenizer = @import("tokenizer.zig");

const html = @embedFile("web/index.html");

const max_body = 1 << 20;
const max_new_tokens = 1024;
const model_id = "vibezig-llm";
const owner = "vibezig";

pub fn serve(gpa: std.mem.Allocator, io: std.Io, model_path: []const u8, port: u16, out: *std.Io.Writer) !void {
    var m = try model_mod.Model.load(gpa, io, model_path, 512 << 20);
    defer m.deinit();
    var session = try gpt.Session.init(gpa, &m, 1);
    defer session.deinit();

    const addr = try std.Io.net.IpAddress.parse("0.0.0.0", port);
    var listener = try addr.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    try out.print(
        "serving {s} ({d} params) on http://localhost:{d}/v1  model={s}\n",
        .{ model_path, m.paramCount(), port, model_id },
    );
    try out.flush();

    while (true) {
        const st = listener.accept(io) catch |e| {
            try out.print("accept failed: {t}\n", .{e});
            try out.flush();
            continue;
        };
        const ctx = Conn{ .io = io, .gpa = gpa, .model = &m, .session = session, .stream = st, .log = out };
        ctx.handle() catch |e| {
            try out.print("request failed: {t}\n", .{e});
            try out.flush();
        };
        st.close(io);
    }
}

const Conn = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    model: *model_mod.Model,
    session: *gpt.Session,
    stream: std.Io.net.Stream,
    log: *std.Io.Writer,

    fn handle(self: Conn) !void {
        var rbuf: [16 * 1024]u8 = undefined;
        var wbuf: [16 * 1024]u8 = undefined;
        var sr = self.stream.reader(self.io, &rbuf);
        var sw = self.stream.writer(self.io, &wbuf);
        var server = std.http.Server.init(&sr.interface, &sw.interface);
        var req = server.receiveHead() catch return;
        try self.route(&req);
    }

    fn route(self: Conn, req: *std.http.Server.Request) !void {
        const target = req.head.target;
        const qmark = std.mem.indexOfScalar(u8, target, '?');
        const path = if (qmark) |i| target[0..i] else target;

        try self.log.print("{s} {s}\n", .{ @tagName(req.head.method), path });
        try self.log.flush();

        if (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/index.html")) {
            return req.respond(html, .{
                .extra_headers = &.{.{ .name = "content-type", .value = "text/html; charset=utf-8" }},
                .keep_alive = false,
            });
        }
        if (std.mem.eql(u8, path, "/favicon.ico")) {
            return req.respond("", .{ .status = .no_content, .keep_alive = false });
        }
        if (std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/api/health")) {
            return sendJson(req, "{\"ok\":true}");
        }
        if (std.mem.eql(u8, path, "/v1/models")) return self.sendModelList(req);
        if (std.mem.startsWith(u8, path, "/v1/models/")) return self.sendModel(req, path["/v1/models/".len..]);
        if (std.mem.eql(u8, path, "/v1/chat/completions")) {
            if (req.head.method != .POST) return methodNotAllowed(req);
            return handleCompletion(self, req, .chat);
        }
        if (std.mem.eql(u8, path, "/v1/completions")) {
            if (req.head.method != .POST) return methodNotAllowed(req);
            return handleCompletion(self, req, .text);
        }
        if (std.mem.startsWith(u8, path, "/v1/")) {
            return sendError(req, .not_found, "unknown endpoint", "invalid_request_error");
        }
        return req.respond("not found", .{ .status = .not_found, .keep_alive = false });
    }

    fn nowSeconds(self: Conn) i64 {
        const ns = std.Io.Clock.Timestamp.now(self.io, .real).raw.nanoseconds;
        return @intCast(@divFloor(ns, std.time.ns_per_s));
    }

    fn sendModelList(self: Conn, req: *std.http.Server.Request) !void {
        var buf: [768]u8 = undefined;
        const body = try std.fmt.bufPrint(&buf,
            \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":{d},"owned_by":"{s}"}}]}}
        , .{ model_id, self.nowSeconds(), owner });
        return sendJson(req, body);
    }

    fn sendModel(self: Conn, req: *std.http.Server.Request, id: []const u8) !void {
        if (!std.mem.eql(u8, id, model_id)) {
            return sendError(req, .not_found, "model not found", "invalid_request_error");
        }
        var buf: [512]u8 = undefined;
        const body = try std.fmt.bufPrint(&buf,
            \\{{"id":"{s}","object":"model","created":{d},"owned_by":"{s}","meta":{{"params":{d},"layers":{d},"heads":{d},"embd":{d},"block":{d},"vocab":{d}}}}}
        , .{
            model_id,
            self.nowSeconds(),
            owner,
            self.model.paramCount(),
            self.model.cfg.n_layer,
            self.model.cfg.n_head,
            self.model.cfg.n_embd,
            self.model.cfg.block_size,
            self.model.cfg.vocab_size,
        });
        return sendJson(req, body);
    }
};

const Mode = enum { chat, text };

const Msg = struct { role: []const u8, content: []const u8 };

/// Request body fields we understand. Unknown fields are ignored, which is what
/// OpenAI clients expect.
const Params = struct {
    model: []const u8 = model_id,
    prompt: []const u8 = "",
    messages: []const Msg = &.{},
    max_tokens: usize = 256,
    temperature: f32 = 0.8,
    top_k: usize = 40,
    seed: u64 = 0,
    stream: bool = false,
};

fn handleCompletion(self: Conn, req: *std.http.Server.Request, mode: Mode) !void {
    var arena_state = std.heap.ArenaAllocator.init(self.gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var body_buf: [8192]u8 = undefined;
    const reader = req.readerExpectNone(&body_buf);
    const body = reader.allocRemaining(arena, .limited(max_body)) catch "";

    var params = Params{};
    parseBody(arena, body, &params) catch |e| {
        return sendError(req, .bad_request, @errorName(e), "invalid_request_error");
    };
    params.max_tokens = @min(@max(params.max_tokens, 1), max_new_tokens);
    params.temperature = @min(@max(params.temperature, 0.0), 2.0);
    params.top_k = @min(params.top_k, 4096);

    const prompt = buildPrompt(arena, &params, mode) catch |e| {
        const msg = switch (e) {
            error.EmptyPrompt => "messages or prompt is required",
            else => "invalid request",
        };
        return sendError(req, .bad_request, msg, "invalid_request_error");
    };

    const tk = tokenizer.Tokenizer.init(self.model.vocab);
    const ids_in = try tk.encode(arena, prompt);
    if (ids_in.len == 0) return sendError(req, .bad_request, "empty prompt", "invalid_request_error");

    if (params.stream) return streamCompletion(self, req, mode, &params, ids_in);
    return fullCompletion(self, req, mode, &params, ids_in);
}

fn parseBody(arena: std.mem.Allocator, body: []const u8, out: *Params) !void {
    if (body.len == 0) return;
    const parsed = try std.json.parseFromSlice(std.json.Value, arena, body, .{});
    if (parsed.value != .object) return error.InvalidJson;
    const obj = parsed.value.object;

    if (obj.get("model")) |v| {
        if (v == .string and v.string.len > 0) out.model = v.string;
    }
    if (obj.get("prompt")) |v| {
        if (v == .string) out.prompt = v.string;
    }
    if (obj.get("stream")) |v| {
        if (v == .bool) out.stream = v.bool;
    }
    if (obj.get("max_tokens") orelse obj.get("max_completion_tokens")) |v| {
        if (jsonInt(v)) |n| out.max_tokens = @intCast(@min(@max(n, 1), @as(i64, max_new_tokens)));
    }
    if (obj.get("temperature")) |v| {
        if (jsonFloat(v)) |f| out.temperature = @floatCast(f);
    }
    if (obj.get("top_k")) |v| {
        if (jsonInt(v)) |n| out.top_k = @intCast(@min(@max(n, 0), @as(i64, 4096)));
    }
    if (obj.get("seed")) |v| {
        if (jsonInt(v)) |n| out.seed = @intCast(@min(@max(n, 0), @as(i64, std.math.maxInt(u32))));
    }
    if (obj.get("messages")) |v| {
        if (v == .array) {
            const msgs = try arena.alloc(Msg, v.array.items.len);
            for (v.array.items, 0..) |item, i| {
                msgs[i] = .{ .role = "user", .content = "" };
                if (item != .object) continue;
                const mo = item.object;
                if (mo.get("role")) |r| {
                    if (r == .string) msgs[i].role = r.string;
                }
                if (mo.get("content")) |c| {
                    switch (c) {
                        .string => msgs[i].content = c.string,
                        // content parts: [{type:"text", text:"..."}]
                        .array => {
                            var buf: std.ArrayList(u8) = .empty;
                            for (c.array.items) |part| {
                                if (part != .object) continue;
                                if (part.object.get("text")) |t| {
                                    if (t == .string) try buf.appendSlice(arena, t.string);
                                }
                            }
                            msgs[i].content = buf.items;
                        },
                        else => {},
                    }
                }
            }
            out.messages = msgs;
        }
    }
}

fn jsonInt(v: std.json.Value) ?i64 {
    return switch (v) {
        .integer => |n| n,
        .float => |f| @intFromFloat(f),
        else => null,
    };
}

fn jsonFloat(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |f| f,
        else => null,
    };
}

/// Chat prompt template: `role: content\n` per message, then `assistant:`.
/// A byte-level model has no chat format of its own, so this is the contract.
fn buildPrompt(arena: std.mem.Allocator, p: *const Params, mode: Mode) ![]const u8 {
    if (mode == .text) {
        if (p.prompt.len == 0) return error.EmptyPrompt;
        return p.prompt;
    }
    if (p.messages.len == 0) {
        if (p.prompt.len > 0) return p.prompt;
        return error.EmptyPrompt;
    }
    var list: std.ArrayList(u8) = .empty;
    for (p.messages) |m| {
        try list.appendSlice(arena, m.role);
        try list.appendSlice(arena, ": ");
        try list.appendSlice(arena, m.content);
        try list.append(arena, '\n');
    }
    try list.appendSlice(arena, "assistant:");
    return list.items;
}

fn fullCompletion(self: Conn, req: *std.http.Server.Request, mode: Mode, params: *const Params, prompt_ids: []const u32) !void {
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(self.gpa);

    self.session.generate(self.gpa, prompt_ids, params.max_tokens, .{
        .temperature = params.temperature,
        .top_k = params.top_k,
        .seed = params.seed,
    }, null, &ids) catch |e| {
        return sendError(req, .internal_server_error, @errorName(e), "server_error");
    };

    const text = try decodeBytes(self.gpa, self.model.vocab, ids.items);
    defer self.gpa.free(text);

    const created = nowSeconds(self.io);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(self.gpa);

    const c: usize = ids.items.len;
    const p: usize = prompt_ids.len;

    if (mode == .chat) {
        try out.print(self.gpa, "{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":", .{ created, created });
        try appendJsonString(&out, self.gpa, params.model);
        try out.appendSlice(self.gpa, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        try appendJsonString(&out, self.gpa, text);
        try out.appendSlice(self.gpa, "},\"logprobs\":null,\"finish_reason\":\"length\"}],\"usage\":");
    } else {
        try out.print(self.gpa, "{{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{ created, created });
        try appendJsonString(&out, self.gpa, params.model);
        try out.appendSlice(self.gpa, ",\"choices\":[{\"index\":0,\"text\":");
        try appendJsonString(&out, self.gpa, text);
        try out.appendSlice(self.gpa, ",\"logprobs\":null,\"finish_reason\":\"length\"}],\"usage\":");
    }
    try out.print(self.gpa, "{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}", .{ p, c, p + c });

    return sendJson(req, out.items);
}

fn nowSeconds(io: std.Io) i64 {
    const ns = std.Io.Clock.Timestamp.now(io, .real).raw.nanoseconds;
    return @intCast(@divFloor(ns, std.time.ns_per_s));
}

fn streamCompletion(self: Conn, req: *std.http.Server.Request, mode: Mode, params: *const Params, prompt_ids: []const u32) !void {
    var body_buf: [8192]u8 = undefined;
    var bw = try req.respondStreaming(&body_buf, .{
        .respond_options = .{
            .keep_alive = false,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "text/event-stream" },
                .{ .name = "cache-control", .value = "no-cache" },
            },
        },
    });
    const w = &bw.writer;

    var sse = SseOut.init(w, params.model, mode, nowSeconds(self.io));
    try sse.emitRole();

    var ctx = StreamCtx{ .sse = &sse, .vocab = self.model.vocab };
    const sink = gpt.TokenSink{ .ctx = @ptrCast(&ctx), .emit = &StreamCtx.emit };

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(self.gpa);
    self.session.generate(self.gpa, prompt_ids, params.max_tokens, .{
        .temperature = params.temperature,
        .top_k = params.top_k,
        .seed = params.seed,
    }, sink, &ids) catch |e| {
        try sse.emitError(@errorName(e));
    };
    try sse.emitFinish("length");
    try sse.emitDone();
    try w.flush();
    try bw.end();
}

/// Emits OpenAI-style SSE frames. Chat mode uses `delta`, text mode uses `text`.
const SseOut = struct {
    w: *std.Io.Writer,
    model: []const u8,
    chat: bool,
    created: i64,
    id_buf: [32]u8 = undefined,
    id: []const u8 = "",

    fn init(w: *std.Io.Writer, model: []const u8, mode: Mode, created: i64) SseOut {
        return .{ .w = w, .model = model, .chat = mode == .chat, .created = created };
    }

    fn head(self: *SseOut, kind: []const u8) !void {
        if (self.id.len == 0) {
            const prefix = if (self.chat) "chatcmpl-" else "cmpl-";
            self.id = std.fmt.bufPrint(&self.id_buf, "{s}{d}", .{ prefix, self.created }) catch "cmpl-0";
        }
        try self.w.print("data: {{\"id\":\"{s}\",\"object\":\"{s}\",\"created\":{d},\"model\":", .{ self.id, if (self.chat) "chat.completion.chunk" else "text_completion", self.created });
        try writeJsonEscaped(self.w, self.model);
        if (self.chat) {
            try self.w.writeAll(",\"choices\":[{\"index\":0,\"delta\":");
        } else {
            try self.w.writeAll(",\"choices\":[{\"index\":0,\"text\":");
        }
        _ = kind;
    }

    fn emitRole(self: *SseOut) !void {
        if (!self.chat) return;
        try self.head("role");
        try self.w.writeAll("{\"role\":\"assistant\",\"content\":\"\"}");
        try self.w.writeAll(",\"logprobs\":null,\"finish_reason\":null}]}\n\n");
        try self.w.flush();
    }

    fn emitContent(self: *SseOut, bytes: []const u8) !void {
        try self.head("content");
        if (self.chat) try self.w.writeAll("{\"content\":") else try self.w.writeAll("\"");
        if (self.chat) {
            try writeJsonEscaped(self.w, bytes);
            try self.w.writeAll("}");
        } else {
            try writeJsonEscaped(self.w, bytes);
            try self.w.writeAll("\"");
        }
        try self.w.writeAll(",\"logprobs\":null,\"finish_reason\":null}]}\n\n");
        try self.w.flush();
    }

    fn emitFinish(self: *SseOut, reason: []const u8) !void {
        try self.head("finish");
        if (self.chat) try self.w.writeAll("{}") else try self.w.writeAll("\"\"");
        try self.w.print(",\"logprobs\":null,\"finish_reason\":\"{s}\"}}]}}\n\n", .{reason});
        try self.w.flush();
    }

    fn emitError(self: *SseOut, msg: []const u8) !void {
        try self.w.print("data: {{\"error\":{{\"message\":\"{s}\",\"type\":\"server_error\"}}}}\n\n", .{msg});
        try self.w.flush();
    }

    fn emitDone(self: *SseOut) !void {
        try self.w.writeAll(DONE);
        try self.w.writeByte(10);
        try self.w.writeByte(10);
        try self.w.flush();
    }
};

const StreamCtx = struct {
    sse: *SseOut,
    vocab: []const u8,
    pending: [4]u8 = undefined,
    len: u8 = 0,
    need: u8 = 0,

    fn emit(ctx: *anyopaque, token: u32) void {
        const self: *StreamCtx = @ptrCast(@alignCast(ctx));
        const b: u8 = if (token < self.vocab.len) self.vocab[token] else '?';
        var out: [4]u8 = undefined;
        if (self.feed(b, &out)) |seq| self.sse.emitContent(seq) catch {};
    }

    fn feed(self: *StreamCtx, b: u8, out: *[4]u8) ?[]const u8 {
        if (self.need == 0) {
            if (b < 0x80) {
                out[0] = b;
                return out[0..1];
            }
            const n: u8 = if (b >= 0xC2 and b <= 0xDF) 2 else if (b >= 0xE0 and b <= 0xEF) 3 else if (b >= 0xF0 and b <= 0xF4) 4 else 0;
            if (n == 0) {
                _ = bad(out);
                return out[0..3];
            }
            self.pending[0] = b;
            self.len = 1;
            self.need = n;
            return null;
        }
        if (b < 0x80 or b > 0xBF) {
            self.len = 0;
            self.need = 0;
            _ = bad(out);
            return out[0..3];
        }
        self.pending[self.len] = b;
        self.len += 1;
        if (self.len < self.need) return null;
        const n2 = self.len;
        @memcpy(out[0..n2], self.pending[0..n2]);
        self.len = 0;
        self.need = 0;
        return out[0..n2];
    }
};

fn bad(out: *[4]u8) [3]u8 {
    out[0] = 0xEF;
    out[1] = 0xBF;
    out[2] = 0xBD;
    return .{ 0xEF, 0xBF, 0xBD };
}

fn writeJsonEscaped(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |b| switch (b) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        10 => try w.writeAll("\\n"),
        13 => try w.writeAll("\\r"),
        9 => try w.writeAll("\\t"),
        else => if (b < 0x20) try w.print("\\u{x:0>4}", .{b}) else try w.writeByte(b),
    };
    try w.writeByte('"');
}

fn appendJsonString(list: *std.ArrayList(u8), gpa: std.mem.Allocator, s: []const u8) !void {
    try list.append(gpa, '"');
    for (s) |b| {
        var tmp: [8]u8 = undefined;
        try list.appendSlice(gpa, escapeByte(&tmp, b));
    }
    try list.append(gpa, '"');
}

fn escapeByte(buf: *[8]u8, b: u8) []const u8 {
    return switch (b) {
        '"' => "\\\"",
        '\\' => "\\\\",
        10 => "\\n",
        13 => "\\r",
        9 => "\\t",
        else => blk: {
            if (b < 0x20) break :blk std.fmt.bufPrint(buf, "\\u{x:0>4}", .{b}) catch "?";
            buf[0] = b;
            break :blk buf[0..1];
        },
    };
}

fn decodeBytes(gpa: std.mem.Allocator, vocab: []const u8, ids: []const u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    const repl = "\xEF\xBF\xBD";
    var pend: [4]u8 = undefined;
    var len: u8 = 0;
    var need: u8 = 0;
    for (ids) |id| {
        const b: u8 = if (id < vocab.len) vocab[id] else '?';
        if (need == 0) {
            if (b < 0x80) {
                try out.append(gpa, b);
                continue;
            }
            const n: u8 = if (b >= 0xC2 and b <= 0xDF) 2 else if (b >= 0xE0 and b <= 0xEF) 3 else if (b >= 0xF0 and b <= 0xF4) 4 else 0;
            if (n == 0) {
                try out.appendSlice(gpa, repl);
                continue;
            }
            pend[0] = b;
            len = 1;
            need = n;
            continue;
        }
        if (b < 0x80 or b > 0xBF) {
            try out.appendSlice(gpa, repl);
            len = 0;
            need = 0;
            continue;
        }
        pend[len] = b;
        len += 1;
        if (len < need) continue;
        try out.appendSlice(gpa, pend[0..len]);
        len = 0;
        need = 0;
    }
    if (len > 0) try out.appendSlice(gpa, repl);
    return out.toOwnedSlice(gpa);
}

fn sendJson(req: *std.http.Server.Request, body: []const u8) !void {
    try req.respond(body, .{
        .extra_headers = &.{
            .{ .name = "content-type", .value = "application/json" },
            .{ .name = "access-control-allow-origin", .value = "*" },
        },
        .keep_alive = false,
    });
}

fn sendError(req: *std.http.Server.Request, status: std.http.Status, msg: []const u8, kind: []const u8) !void {
    var buf: [512]u8 = undefined;
    const body = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\",\"code\":{d}}}}}", .{ msg, kind, @intFromEnum(status) }) catch "{\"error\":{}}";
    try req.respond(body, .{
        .status = status,
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
        .keep_alive = false,
    });
}

fn methodNotAllowed(req: *std.http.Server.Request) !void {
    return sendError(req, .method_not_allowed, "method not allowed", "invalid_request_error");
}
const DONE = "data: " ++ "[" ++ "DONE" ++ "]";

test "parseBody reads messages and max_tokens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var p = Params{};
    try parseBody(arena.allocator(), "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}", &p);
    try std.testing.expectEqual(@as(usize, 8), p.max_tokens);
    try std.testing.expectEqual(@as(usize, 1), p.messages.len);
    try std.testing.expectEqualStrings("hi", p.messages[0].content);
}

test "buildPrompt renders the chat template" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const msgs = [_]Msg{.{ .role = "user", .content = "hi" }};
    var p = Params{ .messages = &msgs };
    const out = try buildPrompt(arena.allocator(), &p, .chat);
    try std.testing.expect(std.mem.startsWith(u8, out, "user: hi"));
    try std.testing.expect(std.mem.endsWith(u8, out, "assistant:"));
}
