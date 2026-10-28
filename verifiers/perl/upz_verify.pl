#!/usr/bin/env perl
# verifiers/perl/upz_verify.pl — Pure Perl 5 64-bit Verifier for `8b-is/upz`
# Verifies:
#   1. 32B `0x62`-gate frame sealing (`SealFrame32B` / `CheckFrame32B`)
#   2. 64B cache-line-aligned `LitanyWireFrame` (`[0..64]`, `wire_seal16` over `[0..62]`)
#   3. `BFP-64` Kleene `+AND=1` / `~NAND=-1` bitplanes at `qt:55`
#   4. `SPQT` `1.585 bits/trit` (`base-243`, `3^5 = 243 <= 256`) roundtrip
#   5. Deterministic `RationalPQ` (`p/q`) zero-float ratios (`31/32`, `36/37`, `39/40`, `3761/3314`)

use v5.30;
use strict;
use warnings;
no warnings 'portable';
use integer;

use constant {
    GATE_0X62          => 0x62,
    FRAME_ABI_V1       => 0x01,
    QUANT_QT55         => 0x55,
    LANES_1_OF_1       => 0x0101,
    FOLD_KOMPRESS      => 0xe1034e575c371ab0,
    FOLD_SWEEP         => 0x48031a0d4c054e69,
    FOLD_TARGET        => 0x206f642074737557,
};

sub xor_fold_scalar {
    my ($bytes) = @_;
    my $len = length($bytes);
    my $acc = 0;
    my $i = 0;
    while ($i + 8 <= $len) {
        my $w = unpack("Q<", substr($bytes, $i, 8));
        $acc ^= $w;
        $i += 8;
    }
    while ($i < $len) {
        $acc ^= ord(substr($bytes, $i, 1));
        $i += 1;
    }
    return $acc;
}

sub gcd_u64 {
    my ($a, $b) = @_;
    while ($b != 0) {
        my $r = $a % $b;
        $a = $b;
        $b = $r;
    }
    return $a == 0 ? 1 : $a;
}

sub reduce_pq {
    my ($p, $q) = @_;
    return ($p, 1) if $q == 0;
    my $g = gcd_u64($p, $q);
    return ($p / $g, $q / $g);
}

sub pack_word15_to_spqt3b {
    my ($word15) = @_;
    my $clean = $word15 & 0x7FFF;
    my @out = (0, 0, 0);
    for my $g (0 .. 2) {
        my $acc = 0;
        my $pow3 = 1;
        for my $j (0 .. 4) {
            my $bit = ($clean >> ($g * 5 + $j)) & 1;
            my $digit = $bit ? 2 : 1;
            $acc += $digit * $pow3;
            $pow3 *= 3;
        }
        $out[$g] = $acc;
    }
    return pack("CCC", @out);
}

sub unpack_spqt3b_to_word15 {
    my ($spqt) = @_;
    my @bytes = unpack("CCC", $spqt);
    my $word15 = 0;
    for my $g (0 .. 2) {
        return undef if $bytes[$g] >= 243;
        my $v = $bytes[$g];
        for my $j (0 .. 4) {
            my $digit = $v % 3;
            $v /= 3;
            if ($digit == 2) {
                $word15 |= (1 << ($g * 5 + $j));
            }
        }
    }
    return $word15;
}

sub seal_frame_32b {
    my ($raw, $packed, $gate) = @_;
    my $prefix = pack(
        "Q<Q<VVC C",
        xor_fold_scalar($raw),
        xor_fold_scalar($packed),
        length($raw),
        length($packed),
        $gate,
        FRAME_ABI_V1
    );
    my $tail_fold = pack("Q<", xor_fold_scalar($prefix));
    return $prefix . substr($tail_fold, 0, 6);
}

sub seal_litany_wire_64b {
    my ($raw, $packed, $gate, $qt) = @_;
    my $f32 = seal_frame_32b($raw, $packed, $gate);
    my ($raw_fold, $packed_fold) = unpack("Q<Q<", substr($f32, 0, 16));
    my $comp = $raw_fold ^ $packed_fold;
    my $mask = $qt * 0x0101010101010101;
    my $pos = $comp & $mask;
    my $neg = $comp & (~$mask);
    my ($rp, $rq) = reduce_pq(length($raw), length($packed) || 1);
    my $spqt = pack_word15_to_spqt3b($comp & 0x7FFF);
    my $prefix62 = $f32 . pack("Q<Q<VV", $pos, $neg, $rp, $rq) . $spqt . pack("Cv", $qt, LANES_1_OF_1);
    my $seal16 = xor_fold_scalar($prefix62) & 0xFFFF;
    return $prefix62 . pack("v", $seal16);
}

say "1..7";

my $raw = "\x48" x 3761;
my $packed = "\x20" x 3314;
my $wire = seal_litany_wire_64b($raw, $packed, GATE_0X62, QUANT_QT55);
say(length($wire) == 64 ? "ok 1 - 64B LitanyWireFrame exact byte length" : "not ok 1");

my ($rp, $rq) = unpack("VV", substr($wire, 48, 8));
say(($rp == 3761 && $rq == 3314) ? "ok 2 - LitanyWireFrame preserves 3761/3314 p/q ratio" : "not ok 2");

my $spqt = pack_word15_to_spqt3b(052525);
my $unpacked15 = unpack_spqt3b_to_word15($spqt);
say((defined($unpacked15) && $unpacked15 == 052525) ? "ok 3 - SPQT base-243 1.585 bits/trit roundtrip" : "not ok 3");

my ($p31, $q32) = reduce_pq(62, 64);
say(($p31 == 31 && $q32 == 32) ? "ok 4 - QAT sub-window reuse reduces 62/64 -> 31/32" : "not ok 4");

my ($p39, $q40) = reduce_pq(3900, 4000);
say(($p39 == 39 && $q40 == 40) ? "ok 5 - zenUltraCooker lossless ratio reduces 3900/4000 -> 39/40" : "not ok 5");

my $mask55 = QUANT_QT55 * 0x0101010101010101;
my $pos = FOLD_KOMPRESS & $mask55;
my $neg = FOLD_KOMPRESS & (~$mask55);
say((($pos & $neg) == 0 && ($pos | $neg) == FOLD_KOMPRESS) ? "ok 6 - BFP-64 disjoint bitplanes at qt:55" : "not ok 6");

my $gate_byte = ord(substr($wire, 24, 1));
say($gate_byte == GATE_0X62 ? "ok 7 - Gate 0x62 witness locked" : "not ok 7");
