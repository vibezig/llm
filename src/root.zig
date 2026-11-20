//! vibezig/llm — a fullstack LLM engine written in Zig.
//!
//! Public surface:
//!   * `gpt`      — transformer training/inference/sampling on Fucina
//!   * `rng`      — deterministic PRNG helpers
//!   * `model`    — model config, tensor registry, `.vzm` checkpoint format
//!   * `tokenizer`— byte-level char tokenizer bundled with the model
//!   * `server`   — HTTP API + embedded web UI
pub const rng = @import("rng.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const model = @import("model.zig");
pub const gpt = @import("gpt.zig");
pub const server = @import("server.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
