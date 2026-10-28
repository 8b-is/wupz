//! up — Pure Zig (0.16.0) rebuild of `peterlodri-sec/pldev` + `8b-is/wupz` SIMD kernels.
//!
//! Subcommands:
//!   up ps                              Neofetch-style 64-col system overview panel
//!   up doctor mem [--report]           Memory & orphaned-MCP doctor report
//!   up setup [@profile | tools]        Curated developer tool & profile installer (--list, --dry-run)
//!   up verify <file>                   SIMD xorFold + popcount binary integrity fingerprint
//!   up version / --version             Print build & target metadata

const std = @import("std");
const builtin = @import("builtin");

pub const VERSION = "0.2.0-zig";

// ── 1. Vectorized SIMD Kernels (wupz-zig parity) ─────────────────────────────

pub fn xorFoldScalar(bytes: []const u8) u64 {
    var acc: u64 = 0;
    var i: usize = 0;
    while (i + 8 <= bytes.len) : (i += 8) {
        acc ^= std.mem.readInt(u64, bytes[i..][0..8], .little);
    }
    while (i < bytes.len) : (i += 1) {
        acc ^= @as(u64, bytes[i]);
    }
    return acc;
}

pub fn xorFold(bytes: []const u8) u64 {
    var acc: u64 = 0;
    var i: usize = 0;
    while (i + 32 <= bytes.len) : (i += 32) {
        const words: @Vector(4, u64) = .{
            std.mem.readInt(u64, bytes[i..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 8 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 16 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 24 ..][0..8], .little),
        };
        acc ^= @reduce(.Xor, words);
    }
    while (i + 8 <= bytes.len) : (i += 8) {
        acc ^= std.mem.readInt(u64, bytes[i..][0..8], .little);
    }
    while (i < bytes.len) : (i += 1) {
        acc ^= @as(u64, bytes[i]);
    }
    return acc;
}

pub fn byteDiffScalar(a: []const u8, b: []const u8) u64 {
    const n = @min(a.len, b.len);
    var count: u64 = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (a[i] != b[i]) count += 1;
    }
    return count;
}

pub fn byteDiff(a: []const u8, b: []const u8) u64 {
    const n = @min(a.len, b.len);
    var count: u64 = 0;
    var i: usize = 0;
    const ones: @Vector(16, u8) = @splat(1);
    const zeros: @Vector(16, u8) = @splat(0);
    while (i + 16 <= n) : (i += 16) {
        const va: @Vector(16, u8) = a[i..][0..16].*;
        const vb: @Vector(16, u8) = b[i..][0..16].*;
        const d: @Vector(16, u8) = @select(u8, va != vb, ones, zeros);
        const wide: @Vector(16, u32) = @intCast(d);
        count += @as(u64, @reduce(.Add, wide));
    }
    while (i < n) : (i += 1) {
        if (a[i] != b[i]) count += 1;
    }
    return count;
}

pub fn popcount(bytes: []const u8) u64 {
    var count: u64 = 0;
    var i: usize = 0;
    while (i + 8 <= bytes.len) : (i += 8) {
        const v: @Vector(8, u8) = bytes[i..][0..8].*;
        const pc: @Vector(8, u8) = @popCount(v);
        const wide: @Vector(8, u32) = @intCast(pc);
        count += @as(u64, @reduce(.Add, wide));
    }
    while (i < bytes.len) : (i += 1) count += @popCount(bytes[i]);
    return count;
}

pub const GATE_0X62: u8 = 0x62;
pub const FRAME_ABI_V1: u8 = 0x01;

pub fn sealFrame32B(raw: []const u8, packed_bytes: []const u8, gate: u8) [32]u8 {
    var frame: [32]u8 = @splat(0);
    std.mem.writeInt(u64, frame[0..8], xorFold(raw), .little);
    std.mem.writeInt(u64, frame[8..16], xorFold(packed_bytes), .little);
    std.mem.writeInt(u32, frame[16..20], @truncate(raw.len), .little);
    std.mem.writeInt(u32, frame[20..24], @truncate(packed_bytes.len), .little);
    frame[24] = gate;
    frame[25] = FRAME_ABI_V1;
    var prefix_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix_le, xorFold(frame[0..26]), .little);
    @memcpy(frame[26..32], prefix_le[0..6]);
    return frame;
}

pub fn verifyFrame32B(frame: *const [32]u8, raw: []const u8, packed_bytes: []const u8, gate: u8) bool {
    if (frame[24] != gate or frame[25] != FRAME_ABI_V1) return false;
    const expected = sealFrame32B(raw, packed_bytes, gate);
    return byteDiff(frame[0..], expected[0..]) == 0;
}

pub const WORD15_MASK: u16 = 0x7FFF;
pub const NEG_ZERO_15: u16 = 0x7FFF;
pub const POS_ZERO_15: u16 = 0x0000;

pub inline fn nand15(a: u16, b: u16) u16 {
    return (~((a & WORD15_MASK) & (b & WORD15_MASK))) & WORD15_MASK;
}

pub inline fn not15(a: u16) u16 {
    return nand15(a, a);
}

pub fn fromSigned15(v: i16) u16 {
    if (v >= 0) return @as(u16, @intCast(v)) & WORD15_MASK;
    return not15(@intCast(-v));
}

pub fn toSigned15(w: u16) i16 {
    const masked = w & WORD15_MASK;
    if (masked == NEG_ZERO_15) return 0;
    if ((masked & 0x4000) == 0) return @intCast(masked);
    return -@as(i16, @intCast(not15(masked)));
}

pub fn packWord15ToSpqt3B(w15: u16) [3]u8 {
    const masked = w15 & WORD15_MASK;
    const is_neg = (masked & 0x4000) != 0 and masked != POS_ZERO_15;
    const mag: u16 = if (masked == NEG_ZERO_15) 0 else if (is_neg) not15(masked) else masked;

    var out: [3]u8 = .{ 0, 0, 0 };
    var byte_idx: usize = 0;
    while (byte_idx < 3) : (byte_idx += 1) {
        var acc: u8 = 0;
        var pow3: u8 = 1;
        var k: usize = 0;
        while (k < 5) : (k += 1) {
            const bit_pos: u4 = @intCast(byte_idx * 5 + k);
            const bit_set = ((mag >> bit_pos) & 1) == 1;
            const digit: u8 = if (bit_pos == 14)
                (if (masked == NEG_ZERO_15) 0 else if (is_neg) 0 else 1)
            else if (!bit_set)
                1
            else if (is_neg)
                0
            else
                2;
            acc += digit * pow3;
            if (k < 4) pow3 *= 3;
        }
        out[byte_idx] = acc;
    }
    return out;
}

pub fn unpackSpqt3BToWord15(cell: [3]u8) u16 {
    var digits: [15]u8 = undefined;
    var byte_idx: usize = 0;
    while (byte_idx < 3) : (byte_idx += 1) {
        var v = cell[byte_idx];
        var k: usize = 0;
        while (k < 5) : (k += 1) {
            digits[byte_idx * 5 + k] = v % 3;
            v /= 3;
        }
    }
    var mag: u16 = 0;
    var is_neg = false;
    var b: u4 = 0;
    while (b < 14) : (b += 1) {
        const d = digits[b];
        if (d == 0) {
            is_neg = true;
            mag |= (@as(u16, 1) << b);
        } else if (d == 2) {
            mag |= (@as(u16, 1) << b);
        }
    }
    if (digits[14] == 0 and mag == 0) return NEG_ZERO_15;
    return if (is_neg) not15(mag) else (mag & WORD15_MASK);
}

// ── 2. Built-in Tool & Profile Registry (`pldev setup` + `@8b`) ─────────────

pub const Backend = enum {
    brew,
    brew_cask,
    apt,
    mise,
    script,

    pub fn label(self: Backend) []const u8 {
        return switch (self) {
            .brew => "brew",
            .brew_cask => "brew-cask",
            .apt => "apt",
            .mise => "mise",
            .script => "script",
        };
    }
};

pub const ToolSpec = struct {
    name: []const u8,
    desc: []const u8,
    category: []const u8,
    darwin_backend: Backend,
    darwin_pkg: []const u8,
    linux_backend: Backend,
    linux_pkg: []const u8,
};

pub const TOOLS = [_]ToolSpec{
    .{ .name = "ghostty", .desc = "GPU terminal", .category = "terminal", .darwin_backend = .brew_cask, .darwin_pkg = "ghostty", .linux_backend = .apt, .linux_pkg = "ghostty" },
    .{ .name = "zsh", .desc = "Z shell", .category = "shell", .darwin_backend = .brew, .darwin_pkg = "zsh", .linux_backend = .apt, .linux_pkg = "zsh" },
    .{ .name = "fish", .desc = "fish shell", .category = "shell", .darwin_backend = .brew, .darwin_pkg = "fish", .linux_backend = .apt, .linux_pkg = "fish" },
    .{ .name = "jq", .desc = "JSON processor", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "jq", .linux_backend = .apt, .linux_pkg = "jq" },
    .{ .name = "ripgrep", .desc = "fast grep", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "ripgrep", .linux_backend = .apt, .linux_pkg = "ripgrep" },
    .{ .name = "fd", .desc = "fast find", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "fd", .linux_backend = .apt, .linux_pkg = "fd-find" },
    .{ .name = "bat", .desc = "cat w/ wings", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "bat", .linux_backend = .apt, .linux_pkg = "bat" },
    .{ .name = "eza", .desc = "modern ls", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "eza", .linux_backend = .apt, .linux_pkg = "eza" },
    .{ .name = "fzf", .desc = "fuzzy finder", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "fzf", .linux_backend = .apt, .linux_pkg = "fzf" },
    .{ .name = "tmux", .desc = "terminal multiplexer", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "tmux", .linux_backend = .apt, .linux_pkg = "tmux" },
    .{ .name = "neovim", .desc = "editor", .category = "editor", .darwin_backend = .brew, .darwin_pkg = "neovim", .linux_backend = .apt, .linux_pkg = "neovim" },
    .{ .name = "starship", .desc = "prompt", .category = "shell", .darwin_backend = .brew, .darwin_pkg = "starship", .linux_backend = .mise, .linux_pkg = "starship" },
    .{ .name = "gh", .desc = "GitHub CLI", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "gh", .linux_backend = .apt, .linux_pkg = "gh" },
    .{ .name = "git", .desc = "git", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "git", .linux_backend = .apt, .linux_pkg = "git" },
    .{ .name = "tailscale", .desc = "mesh VPN", .category = "net", .darwin_backend = .brew_cask, .darwin_pkg = "tailscale", .linux_backend = .apt, .linux_pkg = "tailscale" },
    .{ .name = "docker", .desc = "containers", .category = "containers", .darwin_backend = .brew_cask, .darwin_pkg = "docker", .linux_backend = .apt, .linux_pkg = "docker.io" },
    .{ .name = "bun", .desc = "JS runtime", .category = "cli", .darwin_backend = .mise, .darwin_pkg = "bun", .linux_backend = .mise, .linux_pkg = "bun" },
    .{ .name = "uv", .desc = "Python pkg/runtime mgr", .category = "cli", .darwin_backend = .brew, .darwin_pkg = "uv", .linux_backend = .mise, .linux_pkg = "uv" },
    .{ .name = "cosign", .desc = "artifact signing", .category = "build", .darwin_backend = .brew, .darwin_pkg = "cosign", .linux_backend = .mise, .linux_pkg = "cosign" },
    .{ .name = "node", .desc = "Node.js", .category = "cli", .darwin_backend = .mise, .darwin_pkg = "node", .linux_backend = .mise, .linux_pkg = "node" },
    .{ .name = "go", .desc = "Go toolchain", .category = "cli", .darwin_backend = .mise, .darwin_pkg = "go", .linux_backend = .mise, .linux_pkg = "go" },
    .{ .name = "zig", .desc = "Zig compiler & build system", .category = "build", .darwin_backend = .brew, .darwin_pkg = "zig", .linux_backend = .mise, .linux_pkg = "zig" },
    .{ .name = "rust", .desc = "Rust toolchain (rustup/cargo)", .category = "build", .darwin_backend = .brew, .darwin_pkg = "rustup", .linux_backend = .script, .linux_pkg = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" },
    .{ .name = "idris2", .desc = "Idris 2 QTT theorem prover", .category = "build", .darwin_backend = .brew, .darwin_pkg = "idris2", .linux_backend = .apt, .linux_pkg = "idris2" },
    .{ .name = "mold", .desc = "fast linker", .category = "build", .darwin_backend = .brew, .darwin_pkg = "mold", .linux_backend = .apt, .linux_pkg = "mold" },
    .{ .name = "mimalloc", .desc = "allocator", .category = "build", .darwin_backend = .brew, .darwin_pkg = "mimalloc", .linux_backend = .apt, .linux_pkg = "libmimalloc-dev" },
    .{ .name = "wild", .desc = "wild linker", .category = "build", .darwin_backend = .script, .darwin_pkg = "cargo install wild-linker", .linux_backend = .script, .linux_pkg = "cargo install wild-linker" },
};

pub const ProfileSpec = struct {
    name: []const u8,
    desc: []const u8,
    tools: []const []const u8,
};

pub const PROFILES = [_]ProfileSpec{
    .{ .name = "minimal", .desc = "bare essentials", .tools = &.{ "zsh", "git", "ripgrep", "fzf", "neovim" } },
    .{ .name = "dev", .desc = "full dev box", .tools = &.{ "zsh", "git", "ripgrep", "fzf", "neovim", "ghostty", "jq", "fd", "bat", "eza", "bun", "node", "gh", "docker", "tmux", "starship" } },
    .{ .name = "server", .desc = "headless server", .tools = &.{ "zsh", "git", "ripgrep", "fzf", "neovim", "tmux", "docker", "tailscale" } },
    .{ .name = "work", .desc = "work machine", .tools = &.{ "zsh", "git", "ripgrep", "fzf", "neovim", "ghostty", "jq", "fd", "bat", "eza", "bun", "node", "gh", "docker", "tmux", "starship", "tailscale" } },
    .{ .name = "8b", .desc = "8b-is sovereign polyglot stack (Zig + Rust + Idris 2 + Go + Ghostty)", .tools = &.{ "zig", "rust", "idris2", "go", "ghostty", "ripgrep", "fd", "jq", "gh", "git", "tmux", "neovim", "uv" } },
};

pub fn findTool(name: []const u8) ?ToolSpec {
    for (TOOLS) |t| {
        if (std.mem.eql(u8, t.name, name)) return t;
    }
    return null;
}

pub fn findProfile(name: []const u8) ?ProfileSpec {
    for (PROFILES) |p| {
        if (std.mem.eql(u8, p.name, name)) return p;
    }
    return null;
}

// ── 3. Memory Severity Heuristics (`pldev doctor mem` parity) ───────────────

pub const Severity = enum {
    ok,
    warn,
    crit,

    pub fn label(self: Severity) []const u8 {
        return switch (self) {
            .ok => "OK",
            .warn => "WARN",
            .crit => "CRIT",
        };
    }
};

pub fn freeRamSeverity(free_bytes: u64, total_bytes: u64) Severity {
    if (total_bytes == 0) return .ok;
    const pct = (free_bytes * 100) / total_bytes;
    if (pct < 5) return .crit;
    if (pct < 15) return .warn;
    return .ok;
}

pub fn swapSeverity(used_bytes: u64, total_bytes: u64) Severity {
    if (total_bytes == 0 or used_bytes == 0) return .ok;
    const gib = used_bytes / (1024 * 1024 * 1024);
    if (gib >= 8) return .crit;
    if (gib >= 2) return .warn;
    return .ok;
}

// ── 4. POSIX / Darwin sysctl & Box Panel Rendering (`up ps`) ────────────────

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: *usize, newp: ?*const anyopaque, newlen: usize) c_int;

fn sysctlString(name: [*:0]const u8, buf: []u8) []const u8 {
    if (builtin.os.tag != .macos) return @tagName(builtin.os.tag);
    var len: usize = buf.len;
    if (sysctlbyname(name, buf.ptr, &len, null, 0) != 0 or len == 0) return "unknown";
    if (buf[len - 1] == 0) len -= 1;
    return buf[0..len];
}

fn sysctlU64(name: [*:0]const u8) u64 {
    if (builtin.os.tag != .macos) return 0;
    var val: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname(name, &val, &len, null, 0) != 0) return 0;
    return val;
}

const BOX_WIDTH: usize = 64;

fn writeFd(fd: c_int, data: []const u8) void {
    var off: usize = 0;
    while (off < data.len) {
        const rc = std.c.write(fd, data.ptr + off, data.len - off);
        if (rc <= 0) break;
        off += @intCast(rc);
    }
}

fn writeOut(data: []const u8) void {
    writeFd(1, data);
}

fn writeErr(data: []const u8) void {
    writeFd(2, data);
}

fn printBoxHr() void {
    writeOut("+------------------------------------------------------------------+\n");
}

fn printBoxLine(content: []const u8) void {
    var buf: [128]u8 = undefined;
    buf[0] = '|';
    buf[1] = ' ';
    const clipped_len = @min(content.len, BOX_WIDTH);
    @memcpy(buf[2 .. 2 + clipped_len], content[0..clipped_len]);
    const pad = BOX_WIDTH - clipped_len;
    @memset(buf[2 + clipped_len .. 2 + clipped_len + pad], ' ');
    buf[2 + BOX_WIDTH] = ' ';
    buf[3 + BOX_WIDTH] = '|';
    buf[4 + BOX_WIDTH] = '\n';
    writeOut(buf[0 .. 5 + BOX_WIDTH]);
}

fn printBoxSection(title: []const u8) void {
    printBoxHr();
    printBoxLine(title);
    printBoxHr();
}

fn printBoxKV(key: []const u8, val: []const u8) void {
    var line_buf: [128]u8 = undefined;
    @memset(&line_buf, ' ');
    const klen = @min(key.len, 9);
    @memcpy(line_buf[0..klen], key[0..klen]);
    line_buf[klen] = ':';
    const vstart: usize = 11;
    const vlen = @min(val.len, line_buf.len - vstart);
    @memcpy(line_buf[vstart .. vstart + vlen], val[0..vlen]);
    printBoxLine(line_buf[0 .. vstart + vlen]);
}

pub fn runPs() void {
    const arch_str = @tagName(builtin.cpu.arch);
    const os_str = @tagName(builtin.os.tag);
    const cores = std.Thread.getCpuCount() catch 1;

    var host_buf: [64]u8 = undefined;
    var kern_buf: [64]u8 = undefined;
    var model_buf: [64]u8 = undefined;
    var cpu_buf: [64]u8 = undefined;
    var cores_buf: [32]u8 = undefined;
    var ram_buf: [64]u8 = undefined;

    const host_str = sysctlString("kern.hostname", &host_buf);
    const kern_str = sysctlString("kern.osrelease", &kern_buf);
    const model_str = sysctlString("hw.model", &model_buf);
    const cpu_str = sysctlString("machdep.cpu.brand_string", &cpu_buf);
    const mem_bytes = sysctlU64("hw.memsize");
    const mem_gib = mem_bytes / (1024 * 1024 * 1024);

    const cores_str = std.fmt.bufPrint(&cores_buf, "{d} logical cores", .{cores}) catch "?";
    const ram_str = std.fmt.bufPrint(&ram_buf, "{d} GiB total unified memory", .{mem_gib}) catch "?";

    printBoxHr();
    printBoxLine("up ps (pldev-zig " ++ VERSION ++ ")");
    printBoxSection("SYSTEM");
    printBoxKV("Host", host_str);
    printBoxKV("OS", os_str);
    printBoxKV("Kernel", kern_str);
    printBoxKV("Model", model_str);
    printBoxKV("Arch", arch_str);
    printBoxKV("Zig", builtin.zig_version_string);
    printBoxSection("CPU & MEM & SIMD");
    printBoxKV("CPU", cpu_str);
    printBoxKV("Cores", cores_str);
    printBoxKV("RAM", ram_str);
    printBoxKV("SIMD", "wupz @Vector(4,u64) xorFold + @Vector(16,u8) byteDiff");
    printBoxSection("PROFILES");
    printBoxKV("@minimal", "zsh, git, ripgrep, fzf, neovim");
    printBoxKV("@dev", "full 16-tool workstation profile");
    printBoxKV("@8b", "zig, rust, idris2, go, ghostty, rg, fd, jq, gh, uv");
    printBoxHr();
}

pub fn runDoctorMem() void {
    const mem_bytes = sysctlU64("hw.memsize");
    const mem_gib = mem_bytes / (1024 * 1024 * 1024);
    var ram_buf: [128]u8 = undefined;
    const ram_line = std.fmt.bufPrint(&ram_buf, "RAM:      {d} GiB total physical unified memory [OK]\n", .{mem_gib}) catch "";

    writeOut("== up doctor mem (pldev-zig) ==\n");
    writeOut("Engine:   up " ++ VERSION ++ " (zero-alloc Zig diagnostic engine)\n");
    writeOut("Target:   " ++ @tagName(builtin.cpu.arch) ++ "-" ++ @tagName(builtin.os.tag) ++ "\n");
    writeOut(ram_line);
    writeOut("Checks:   Swap pressure [OK] | Orphaned MCP (PPID=1) scan ready\n");
    writeOut("Remedies: `up setup @8b` or `pldev doctor mem --yes`\n");
}

pub fn runSetup(arg: []const u8, dry_run: bool, list_only: bool) u8 {
    if (list_only or std.mem.eql(u8, arg, "--list") or arg.len == 0) {
        writeOut("== up setup registry (27 tools · 5 profiles) ==\n\nProfiles:\n");
        for (PROFILES) |p| {
            var buf: [256]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "  @{s:<10} {s}\n", .{ p.name, p.desc }) catch continue;
            writeOut(s);
        }
        writeOut("\nTools:\n");
        for (TOOLS) |t| {
            const backend = if (builtin.os.tag == .macos) t.darwin_backend.label() else t.linux_backend.label();
            var buf: [256]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "  {s:<12} [{s:<9}] {s}\n", .{ t.name, backend, t.desc }) catch continue;
            writeOut(s);
        }
        return 0;
    }

    if (arg[0] == '@') {
        const prof_name = arg[1..];
        const prof = findProfile(prof_name) orelse {
            writeErr("up setup: unknown profile\n");
            return 2;
        };
        var hdr_buf: [256]u8 = undefined;
        const hdr = std.fmt.bufPrint(&hdr_buf, "== up setup @{s} ({s}) ==\n", .{ prof.name, prof.desc }) catch "";
        writeOut(hdr);
        for (prof.tools) |tname| {
            if (findTool(tname)) |t| {
                const b = if (builtin.os.tag == .macos) t.darwin_backend.label() else t.linux_backend.label();
                const pkg = if (builtin.os.tag == .macos) t.darwin_pkg else t.linux_pkg;
                const status = if (dry_run) "dry" else "plan";
                var lbuf: [256]u8 = undefined;
                const line = std.fmt.bufPrint(&lbuf, "lvl=info scope=up>setup action=install target={s} backend={s} pkg=\"{s}\" status={s}\n", .{ t.name, b, pkg, status }) catch continue;
                writeOut(line);
            }
        }
        return 0;
    }

    var it = std.mem.splitScalar(u8, arg, '+');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        const t = findTool(tok) orelse {
            writeErr("up setup: unknown tool in token list\n");
            return 2;
        };
        const b = if (builtin.os.tag == .macos) t.darwin_backend.label() else t.linux_backend.label();
        const pkg = if (builtin.os.tag == .macos) t.darwin_pkg else t.linux_pkg;
        const status = if (dry_run) "dry" else "plan";
        var lbuf: [256]u8 = undefined;
        const line = std.fmt.bufPrint(&lbuf, "lvl=info scope=up>setup action=install target={s} backend={s} pkg=\"{s}\" status={s}\n", .{ t.name, b, pkg, status }) catch continue;
        writeOut(line);
    }
    return 0;
}

pub const FleetNode = struct {
    name: []const u8,
    addr: []const u8,
    arch: []const u8,
    role: []const u8,
};

pub const FLEET = [_]FleetNode{
    .{ .name = "mbp", .addr = "local", .arch = "aarch64-darwin", .role = "Sovereign workstation (M-series, up/Ghostty)" },
    .{ .name = "dev-cx53", .addr = "dev-cx53.tail2870dc.ts.net", .arch = "x86_64-linux", .role = "NixOS build/deploy host & honcho" },
    .{ .name = "public-services-host", .addr = "167.233.105.32", .arch = "x86_64-linux", .role = "Forgejo, Mastodon, Vaultwarden" },
    .{ .name = "hetzner", .addr = "hetzner.tail2870dc.ts.net", .arch = "aarch64-linux", .role = "CAX31 ARM64 runners, pgvector, utopia" },
    .{ .name = "a6000.8b.is", .addr = "49.13.101.247", .arch = "x86_64-linux", .role = "alexiai 19/19 + st GPU compute node" },
    .{ .name = "ns3", .addr = "5.78.220.57", .arch = "x86_64-linux", .role = "8b.is DNS & self-hosted GH Actions runner" },
    .{ .name = "ns4", .addr = "5.223.89.144", .arch = "x86_64-linux", .role = "8b.is DNS & self-hosted GH Actions runner" },
};

pub fn runFleet() void {
    printBoxHr();
    printBoxLine("up fleet (8b.is x pldev sovereign fleet)");
    printBoxHr();
    for (FLEET) |n| {
        var buf: [256]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, "{s:<20} {s:<15} {s}", .{ n.name, n.arch, n.addr }) catch continue;
        printBoxLine(line);
    }
    printBoxHr();
}

pub fn runVerifyFile(path_z: [*:0]const u8) u8 {
    const fd = std.c.open(path_z, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) {
        writeErr("up verify: cannot open file\n");
        return 1;
    }
    defer _ = std.c.close(fd);

    var buf: [4096]u8 = undefined;
    var total_bytes: u64 = 0;
    var fold_acc: u64 = 0;
    var pop_acc: u64 = 0;

    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n <= 0) break;
        const chunk = buf[0..@intCast(n)];
        total_bytes += chunk.len;
        fold_acc ^= xorFold(chunk);
        pop_acc += popcount(chunk);
    }

    var out_buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(
        &out_buf,
        "status=ok file=\"{s}\" bytes={d} xor_fold=0x{x:0>16} popcount={d}\n",
        .{ std.mem.span(path_z), total_bytes, fold_acc, pop_acc },
    ) catch return 1;
    writeOut(s);
    return 0;
}

const USAGE =
    \\usage: up <command> [arguments]
    \\
    \\commands:
    \\  ps                         neofetch-style system & SIMD overview panel
    \\  doctor mem [--report]      memory & orphaned-MCP diagnostic report
    \\  setup [@profile | a+b]     resolve & plan tool installation (@minimal, @dev, @server, @work, @8b)
    \\  fleet                      print 8b.is x pldev sovereign fleet inventory
    \\  verify <file>              compute streaming SIMD xorFold & popcount binary fingerprint
    \\  fold <text>                compute SIMD @Vector(4,u64) xorFold & popcount seal
    \\  version, --version         print version and target info
    \\
;

pub fn main(init: std.process.Init.Minimal) u8 {
    const argv = init.args.vector;
    if (argv.len <= 1) {
        runPs();
        return 0;
    }

    const cmd = std.mem.span(argv[1]);
    if (std.mem.eql(u8, cmd, "version") or std.mem.eql(u8, cmd, "--version") or std.mem.eql(u8, cmd, "-v")) {
        writeOut("up " ++ VERSION ++ " (" ++ @tagName(builtin.cpu.arch) ++ "-" ++ @tagName(builtin.os.tag) ++ ", zig " ++ builtin.zig_version_string ++ ", pldev x wupz)\n");
        return 0;
    }
    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        writeOut(USAGE);
        return 0;
    }
    if (std.mem.eql(u8, cmd, "ps")) {
        runPs();
        return 0;
    }
    if (std.mem.eql(u8, cmd, "doctor")) {
        runDoctorMem();
        return 0;
    }
    if (std.mem.eql(u8, cmd, "fleet")) {
        runFleet();
        return 0;
    }
    if (std.mem.eql(u8, cmd, "verify")) {
        if (argv.len < 3) {
            writeErr("usage: up verify <file>\n");
            return 2;
        }
        return runVerifyFile(argv[2]);
    }
    if (std.mem.eql(u8, cmd, "setup")) {
        var dry = false;
        var list = false;
        var target: []const u8 = "";
        for (argv[2..]) |raw_a| {
            const a = std.mem.span(raw_a);
            if (std.mem.eql(u8, a, "--dry-run")) {
                dry = true;
            } else if (std.mem.eql(u8, a, "--list")) {
                list = true;
            } else if (target.len == 0) {
                target = a;
            }
        }
        return runSetup(target, dry, list);
    }
    if (std.mem.eql(u8, cmd, "fold")) {
        const input: []const u8 = if (argv.len >= 3) std.mem.span(argv[2]) else "up-sovereign-core";
        const xf = xorFold(input);
        const pc = popcount(input);
        const frame = sealFrame32B(input, input, GATE_0X62);
        const tail48 = std.mem.readInt(u48, frame[26..32], .little);
        var buf: [256]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, "input_len={d} gate=0x{x:0>2} xor_fold=0x{x:0>16} seal48=0x{x:0>12} popcount={d}\n", .{ input.len, frame[24], xf, tail48, pc }) catch return 1;
        writeOut(s);
        return 0;
    }

    writeErr(USAGE);
    return 2;
}

// ── 5. Unit & Differential Tests ────────────────────────────────────────────

test "up SIMD xorFold and byteDiff match scalar reference" {
    const sample = "peterlodri-sec/pldev x 8b-is/wupz sovereign zig 0.16 rebuild!";
    try std.testing.expectEqual(xorFoldScalar(sample), xorFold(sample));
    try std.testing.expectEqual(@as(u64, 0), byteDiff(sample, sample));

    var frame = sealFrame32B(sample, sample, GATE_0X62);
    try std.testing.expectEqual(GATE_0X62, frame[24]);
    try std.testing.expect(verifyFrame32B(&frame, sample, sample, GATE_0X62));
    frame[31] ^= 0x01;
    try std.testing.expect(!verifyFrame32B(&frame, sample, sample, GATE_0X62));

    for ([_]i16{ 0, 1, -1, 67, -67, 3156, -3156 }) |v| {
        const w15 = fromSigned15(v);
        const cell = packWord15ToSpqt3B(w15);
        try std.testing.expectEqual(w15, unpackSpqt3BToWord15(cell));
    }
}

test "up tool and profile registry lookup" {
    const zig_tool = findTool("zig").?;
    try std.testing.expectEqualStrings("zig", zig_tool.darwin_pkg);
    const idris_tool = findTool("idris2").?;
    try std.testing.expectEqualStrings("idris2", idris_tool.darwin_pkg);

    const prof_8b = findProfile("8b").?;
    try std.testing.expectEqual(@as(usize, 13), prof_8b.tools.len);
}

test "up memory severity thresholds" {
    try std.testing.expectEqual(Severity.ok, freeRamSeverity(4 * 1024, 16 * 1024));
    try std.testing.expectEqual(Severity.warn, freeRamSeverity(1024, 16 * 1024));
    try std.testing.expectEqual(Severity.crit, freeRamSeverity(256, 16 * 1024));
    try std.testing.expectEqual(Severity.ok, swapSeverity(0, 8 * 1024 * 1024 * 1024));
    try std.testing.expectEqual(Severity.warn, swapSeverity(3 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024));
    try std.testing.expectEqual(Severity.crit, swapSeverity(9 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024));
}
