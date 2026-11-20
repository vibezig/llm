//! vibezig/llm — a fullstack LLM engine written in Zig.
//!
//! Public surface:
//!   * `tensor`   — dense f32 tensor math and NN primitives
//!   * `rng`      — deterministic PRNG helpers
//!   * `model`    — model config, tensor registry, `.vzm` checkpoint format
//!   * `tokenizer`— byte-level char tokenizer bundled with the model
//!   * `gpt`      — transformer forward/backward, sampling, training loop
//!   * `server`   — HTTP API + embedded web UI
pub const tensor = @import("tensor.zig");
pub const rng = @import("rng.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const model = @import("model.zig");
pub const gpt = @import("gpt.zig");
pub const server = @import("server.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
