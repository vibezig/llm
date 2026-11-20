//! HTTP server: JSON API + a single-file web UI embedded in the binary.
//!
//! Routes:
//!   GET  /                  the chat UI
//!   GET  /api/model         model config as JSON
//!   GET  /api/health        liveness probe
//!   GET  /api/generate      ?prompt=&tokens=&temp=&top_k=  (blocking JSON)
//!   POST /api/generate      {"prompt":..,"tokens":..,"temperature":..,"top_k":..}
//!   GET  /api/stream        same params, SSE token stream (what the UI uses)
const std = @import("std");
const model_mod = @import("model.zig");
const gpt = @import("gpt.zig");
const tokenizer = @import("tokenizer.zig");

const html = @embedFile("web/index.html");

const max_body = 1 << 20;
const max_prompt_len = 4096;
const max_new_tokens = 1024;

pub fn serve(gpa: std.mem.Allocator, io: std.Io, model_path: []const u8, port: u16, out: *std.Io.Writer) !void {
    var m = try model_mod.Model.load(gpa, io, model_path, 512 << 20);
    defer m.deinit();
    var session = try gpt.Session.init(gpa, &m, 1);
    defer session.deinit();

    const addr = try std.Io.net.IpAddress.parse("0.0.0.0", port);
    var listener = try addr.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    try out.print(
        "serving {s} ({d} params) on http://localhost:{d}\n",
        .{ model_path, m.paramCount(), port },
    );
    try out.flush();

    while (true) {
        const stream = listener.accept(io) catch |e| {
            try out.print("accept failed: {t}\n", .{e});
            try out.flush();
            continue;
        };
        const ctx = Conn{
            .io = io,
            .gpa = gpa,
            .model = &m,
            .session = &session,
            .stream = stream,
            .log = out,
        };
        ctx.handle() catch |e| {
            try out.print("request failed: {t}\n", .{e});
            try out.flush();
        };
        stream.close(io);
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
        const query = if (qmark) |i| target[i + 1 ..] else "";

        try self.log.print("{s} {s}\n", .{ @tagName(req.head.method), target });
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
        if (std.mem.eql(u8, path, "/api/health")) {
            return sendJson(req, "{\"ok\":true}");
        }
        if (std.mem.eql(u8, path, "/api/model")) {
            return self.sendModelInfo(req);
        }
        if (std.mem.eql(u8, path, "/api/generate")) {
            return self.handleGenerate(req, query);
        }
        if (std.mem.eql(u8, path, "/api/stream")) {
            return self.handleStream(req, query);
        }
        return req.respond("not found", .{ .status = .not_found, .keep_alive = false });
    }

    fn sendModelInfo(self: Conn, req: *std.http.Server.Request) !void {
        const cfg = self.model.cfg;
        var buf: [512]u8 = undefined;
        const body = try std.fmt.bufPrint(&buf,
            \\{{"params":{d},"layers":{d},"heads":{d},"embd":{d},"block":{d},"vocab":{d},"format":"vzm"}}
        , .{
            self.model.paramCount(),
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd,
            cfg.block_size,
            cfg.vocab_size,
        });
        return sendJson(req, body);
    }

    fn handleGenerate(self: Conn, req: *std.http.Server.Request, query: []const u8) !void {
        var params = GenParams{};
        if (req.head.method == .POST) {
            var body_buf: [8192]u8 = undefined;
            const reader = req.readerExpectNone(&body_buf);
            const body = reader.allocRemaining(self.gpa, .limited(max_body)) catch "";
            defer if (body.len > 0) self.gpa.free(body);
            try params.fromJson(self.gpa, body);
        } else {
            try params.fromQuery(self.gpa, query);
        }

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(self.gpa);
        const t0 = std.Io.Clock.Timestamp.now(self.io, .awake).raw.nanoseconds;
        try self.runInference(&params, null, &ids);
        const ns: i96 = std.Io.Clock.Timestamp.now(self.io, .awake).raw.nanoseconds - t0;

        const decoded = try tokenizer.Tokenizer.init(self.model.vocab).decode(self.gpa, ids.items);
        defer self.gpa.free(decoded);

        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(self.gpa);
        try out.appendSlice(self.gpa, "{\"text\":");
        try appendJsonString(&out, self.gpa, decoded);
        try out.print(self.gpa, ",\"tokens\":{d},\"ms\":{d:.1},\"tok_per_s\":{d:.1}}}", .{
            ids.items.len,
            @as(f64, @floatFromInt(ns)) / 1e6,
            @as(f64, @floatFromInt(ids.items.len)) / @max(@as(f64, @floatFromInt(ns)) / 1e9, 1e-6),
        });
        return sendJson(req, out.items);
    }

    fn handleStream(self: Conn, req: *std.http.Server.Request, query: []const u8) !void {
        var params = GenParams{};
        try params.fromQuery(self.gpa, query);

        var body_buf: [8192]u8 = undefined;
        var bw = try req.respondStreaming(&body_buf, .{
            .respond_options = .{
                .keep_alive = false,
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "text/event-stream" },
                    .{ .name = "cache-control", .value = "no-cache" },
                    .{ .name = "x-accel-buffering", .value = "no" },
                },
            },
        });
        const w = &bw.writer;

        var sse = SseSink{ .w = w, .vocab = self.model.vocab };
        try sse.writeEvent("start", "");
        const sink = gpt.TokenSink{ .ctx = @ptrCast(&sse), .emit = &SseSink.emit };
        self.runInference(&params, sink, null) catch |e| {
            try sse.writeError(@errorName(e));
        };
        try sse.writeEvent("done", "");
        try w.flush();
        try bw.end();
    }

    fn runInference(self: Conn, params: *const GenParams, sink: ?gpt.TokenSink, out: ?*std.ArrayList(u32)) !void {
        const tk = tokenizer.Tokenizer.init(self.model.vocab);
        const prompt = if (params.prompt.len == 0) "\n" else params.prompt;
        if (prompt.len > max_prompt_len) return error.PromptTooLong;
        if (params.tokens > max_new_tokens) return error.TooManyTokens;

        const prompt_ids = try tk.encode(self.gpa, prompt);
        defer self.gpa.free(prompt_ids);

        var local: std.ArrayList(u32) = .empty;
        defer local.deinit(self.gpa);
        const dst = out orelse &local;

        try self.session.generate(self.gpa, prompt_ids, params.tokens, .{
            .temperature = params.temperature,
            .top_k = params.top_k,
            .seed = params.seed,
        }, sink, dst);
    }
};

/// Streams each token as an SSE `data:` line.
const SseSink = struct {
    w: *std.Io.Writer,
    vocab: []const u8,

    /// Tokens are raw bytes, so each event carries the byte value as an
    /// integer; the browser reassembles them with a streaming TextDecoder.
    /// This keeps multi-byte UTF-8 sequences intact across event boundaries.
    fn emit(ctx: *anyopaque, token: u32) void {
        const self: *SseSink = @ptrCast(@alignCast(ctx));
        const b: u8 = if (token < self.vocab.len) self.vocab[token] else '?';
        self.w.print("data: {{\"b\":{d}}}\n\n", .{b}) catch return;
        self.w.flush() catch {};
    }

    fn writeEvent(self: *SseSink, name: []const u8, payload: []const u8) !void {
        try self.w.print("data: {{\"event\":\"{s}\"{s}}}\n\n", .{ name, payload });
        try self.w.flush();
    }

    fn writeError(self: *SseSink, msg: []const u8) !void {
        var buf: [128]u8 = undefined;
        const payload = try std.fmt.bufPrint(&buf, ",\"error\":\"{s}\"", .{msg});
        try self.writeEvent("error", payload);
    }
};

fn escapeByte(buf: *[8]u8, b: u8) []const u8 {
    return switch (b) {
        '"' => "\\\"",
        '\\' => "\\\\",
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
        else => blk: {
            if (b < 0x20) {
                break :blk std.fmt.bufPrint(buf, "\\u{x:0>4}", .{b}) catch "?";
            }
            buf[0] = b;
            break :blk buf[0..1];
        },
    };
}

const GenParams = struct {
    prompt: []const u8 = "",
    tokens: usize = 200,
    temperature: f32 = 0.8,
    top_k: usize = 40,
    seed: u64 = 0,
    /// Backing storage for the prompt so `prompt` never dangles.
    storage: [max_prompt_len + 1]u8 = undefined,

    fn setPrompt(self: *GenParams, raw: []const u8) void {
        self.prompt = percentDecode(raw, &self.storage);
    }

    fn fromQuery(self: *GenParams, gpa: std.mem.Allocator, query: []const u8) !void {
        _ = gpa;
        var it = std.mem.splitScalar(u8, query, '&');
        while (it.next()) |pair| {
            if (pair.len == 0) continue;
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
            const key = pair[0..eq];
            const raw = pair[eq + 1 ..];
            if (std.mem.eql(u8, key, "prompt")) {
                self.setPrompt(raw);
            } else if (std.mem.eql(u8, key, "tokens")) {
                self.tokens = clampInt(raw, 1, max_new_tokens, 200);
            } else if (std.mem.eql(u8, key, "temp")) {
                self.temperature = clampFloat(raw, 0.0, 2.0, 0.8);
            } else if (std.mem.eql(u8, key, "top_k")) {
                self.top_k = clampInt(raw, 0, 4096, 40);
            } else if (std.mem.eql(u8, key, "seed")) {
                self.seed = clampInt(raw, 0, std.math.maxInt(u32), 0);
            }
        }
    }

    fn fromJson(self: *GenParams, gpa: std.mem.Allocator, body: []const u8) !void {
        if (body.len == 0) return;
        const parsed = std.json.parseFromSlice(std.json.Value, gpa, body, .{}) catch return;
        defer parsed.deinit();
        if (parsed.value != .object) return;
        const obj = parsed.value.object;
        if (obj.get("prompt")) |v| {
            if (v == .string) self.setPrompt(v.string);
        }
        if (obj.get("tokens")) |v| {
            if (v == .integer) {
                const clamped = @max(@as(i64, 1), @min(v.integer, @as(i64, max_new_tokens)));
                self.tokens = @intCast(clamped);
            }
        }
        if (obj.get("temperature") orelse obj.get("temp")) |v| {
            if (v == .float) self.temperature = @floatCast(v.float);
            if (v == .integer) self.temperature = @floatFromInt(v.integer);
        }
        if (obj.get("top_k")) |v| {
            if (v == .integer) {
                const clamped = @max(@as(i64, 0), @min(v.integer, 4096));
                self.top_k = @intCast(clamped);
            }
        }
        if (obj.get("seed")) |v| {
            if (v == .integer) {
                const clamped = @max(@as(i64, 0), v.integer);
                self.seed = @intCast(@min(clamped, std.math.maxInt(u32)));
            }
        }
    }
};

fn percentDecode(raw: []const u8, buf: []u8) []const u8 {
    var n: usize = 0;
    var i: usize = 0;
    while (i < raw.len and n < buf.len) : (i += 1) {
        const ch = raw[i];
        if (ch == '%' and i + 2 < raw.len) {
            const hi = std.fmt.charToDigit(raw[i + 1], 16) catch {
                buf[n] = ch;
                n += 1;
                continue;
            };
            const lo = std.fmt.charToDigit(raw[i + 2], 16) catch {
                buf[n] = ch;
                n += 1;
                continue;
            };
            buf[n] = hi * 16 + lo;
            n += 1;
            i += 2;
        } else if (ch == '+') {
            buf[n] = ' ';
            n += 1;
        } else {
            buf[n] = ch;
            n += 1;
        }
    }
    return buf[0..n];
}

fn clampInt(raw: []const u8, lo: usize, hi: usize, default: usize) usize {
    const v = std.fmt.parseInt(usize, raw, 10) catch return default;
    return @min(@max(v, lo), hi);
}

fn clampFloat(raw: []const u8, lo: f32, hi: f32, default: f32) f32 {
    const v = std.fmt.parseFloat(f32, raw) catch return default;
    return @min(@max(v, lo), hi);
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

fn appendJsonString(list: *std.ArrayList(u8), gpa: std.mem.Allocator, s: []const u8) !void {
    try list.append(gpa, '"');
    for (s) |b| {
        var tmp: [8]u8 = undefined;
        try list.appendSlice(gpa, escapeByte(&tmp, b));
    }
    try list.append(gpa, '"');
}

test "percentDecode handles escapes and plus" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("a b&c", percentDecode("a%20b%26c", &buf));
    try std.testing.expectEqualStrings("a b", percentDecode("a+b", &buf));
    try std.testing.expectEqualStrings("100%", percentDecode("100%", &buf));
}

test "json escaping matches the spec subset we emit" {
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(std.testing.allocator);
    try appendJsonString(&list, std.testing.allocator, "a\"b\nc");
    try std.testing.expectEqualStrings("\"a\\\"b\\nc\"", list.items);
}
