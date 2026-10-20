import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeAlp } from '../src/codec/alp.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const VECTOR_SIZE = 1024;
const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);

interface VectorSpec {
  e: number;
  f: number;
  bitWidth: number;
  frame: bigint;
  deltas: bigint[];
  /** Positions (and values) replaced after decoding, as the encoder does for values that do not round-trip. */
  exceptions?: Array<[number, number]>;
}

const pow10 = (n: number) => Number(`1e${n}`);

/** Bit-packs the deltas, least significant bit first, with no padding between them. */
function pack(deltas: bigint[], bitWidth: number): Uint8Array {
  const packed = new Uint8Array(Math.ceil((deltas.length * bitWidth) / 8));
  deltas.forEach((delta, i) => {
    for (let b = 0; b < bitWidth; b++) {
      if ((delta >> BigInt(b)) & 1n) {
        const pos = i * bitWidth + b;
        packed[pos >> 3] |= 1 << (pos & 7);
      }
    }
  });
  return packed;
}

/** Builds a wire-format ALP block (big-endian, as written by the Java encoder). */
function encodeVectors(specs: VectorSpec[]): Uint8Array {
  const parts: Uint8Array[] = [];
  const head = new DataView(new ArrayBuffer(4));
  head.setInt32(0, specs.reduce((n, s) => n + s.deltas.length, 0), false);
  parts.push(new Uint8Array(head.buffer));

  for (const s of specs) {
    const packed = pack(s.deltas, s.bitWidth);
    const exceptions = s.exceptions ?? [];
    const bytes = new Uint8Array(1 + 1 + 1 + 8 + 2 + 2 + packed.length + exceptions.length * 10);
    const view = new DataView(bytes.buffer);
    bytes[0] = s.e;
    bytes[1] = s.f;
    bytes[2] = s.bitWidth;
    view.setBigInt64(3, s.frame, false);
    view.setUint16(11, exceptions.length, false);
    view.setUint16(13, packed.length, false);
    bytes.set(packed, 15);
    let pos = 15 + packed.length;
    for (const [index, value] of exceptions) {
      view.setUint16(pos, index, false);
      view.setFloat64(pos + 2, value, false);
      pos += 10;
    }
    parts.push(bytes);
  }

  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/**
 * What the Java decoder computes: the encoded long is frame + delta (wrapping at 64 bits),
 * rounded once to a double, then scaled.
 */
function expectedValues(s: VectorSpec): number[] {
  const values = s.deltas.map((delta) => {
    const encoded = Number(BigInt.asIntN(64, s.frame + delta));
    return (encoded * pow10(s.f)) / pow10(s.e);
  });
  for (const [index, value] of s.exceptions ?? []) values[index] = value;
  return values;
}

function expectDecodesExactly(specs: VectorSpec[]) {
  const out = decodeAlp(encodeVectors(specs));
  expect(Array.from(out)).toEqual(specs.flatMap(expectedValues));
}

/** Deterministic PRNG so a failure reproduces. */
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomBits(rand: () => number, bits: number): bigint {
  let v = 0n;
  for (let done = 0; done < bits; done += 32) {
    v = (v << 32n) | BigInt(Math.floor(rand() * 4294967296));
  }
  return bits === 0 ? 0n : v & ((1n << BigInt(bits)) - 1n);
}

const randomDeltas = (rand: () => number, count: number, bitWidth: number) =>
  Array.from({ length: count }, () => randomBits(rand, bitWidth));

describe('ALP decoder: deltas wider than 31 bits', () => {
  it('reads a 35-bit delta vector instead of a nearly flat line', () => {
    // Price-like data with many digits: e = 18, frame 0.000435, deltas up to ~1.8e10.
    const spec: VectorSpec = {
      e: 18,
      f: 0,
      bitWidth: 35,
      frame: 435_000_000_000_000n,
      deltas: Array.from({ length: 144 }, (_, i) => BigInt(i) * 123_456_789n),
    };
    expectDecodesExactly([spec]);

    const out = decodeAlp(encodeVectors([spec]));
    expect(out[0]).toBe(0.000435);
    expect(out[143]).toBeCloseTo(0.000435017654320827, 18);
  });

  it('reads 52-bit deltas on the number path', () => {
    const rand = mulberry32(1);
    expectDecodesExactly([
      { e: 0, f: 0, bitWidth: 52, frame: 1_000_000n, deltas: randomDeltas(rand, 200, 52) },
    ]);
  });

  it('reads every width from 1 to 64 bits', () => {
    const rand = mulberry32(2);
    for (let bitWidth = 1; bitWidth <= 64; bitWidth++) {
      expectDecodesExactly([
        { e: 3, f: 1, bitWidth, frame: -123_456n, deltas: randomDeltas(rand, 70, bitWidth) },
      ]);
    }
  });
});

describe('ALP decoder: integers beyond 2^53', () => {
  it('reads a frame larger than 2^53', () => {
    // 63146.xxxxxxxx at e = 13 is ~6.3e17, where doubles are 128 apart: the frame alone is not exact.
    // frame = 63 (mod 128) makes rounding the frame first and the sum second differ from rounding
    // the exact sum once, which is what Java's (double) cast does.
    const rand = mulberry32(3);
    const frame = 631_460_000_000_000_000n - (631_460_000_000_000_000n % 128n) + 63n;
    expectDecodesExactly([
      { e: 13, f: 0, bitWidth: 20, frame, deltas: randomDeltas(rand, VECTOR_SIZE, 20) },
    ]);
  });

  it('reads a negative frame that the deltas carry across zero', () => {
    const rand = mulberry32(4);
    expectDecodesExactly([
      { e: 5, f: 2, bitWidth: 42, frame: -(1n << 40n), deltas: randomDeltas(rand, 200, 42) },
    ]);
  });

  it('is exact on both sides of the number-path limit', () => {
    const rand = mulberry32(5);
    const bitWidth = 20;
    const limit = MAX_SAFE - (1n << BigInt(bitWidth)); // largest frame the number path accepts
    // every vector but the last one holds exactly VECTOR_SIZE values
    const full = [0n, 1n, (1n << 20n) - 1n, ...randomDeltas(rand, VECTOR_SIZE - 3, bitWidth)];
    expectDecodesExactly([
      { e: 4, f: 0, bitWidth, frame: limit, deltas: full },
      { e: 4, f: 0, bitWidth, frame: limit + 1n, deltas: full },
      { e: 4, f: 0, bitWidth, frame: -limit, deltas: full },
      { e: 4, f: 0, bitWidth, frame: -limit - 1n, deltas: full },
    ]);
  });

  it('reads deltas of a full 64 bits', () => {
    expectDecodesExactly([
      {
        e: 0,
        f: 0,
        bitWidth: 64,
        frame: -(1n << 63n),
        deltas: [0n, 1n, 1n << 63n, (1n << 63n) + 5n, (1n << 64n) - 1n],
      },
    ]);
  });
});

describe('ALP decoder: per-vector decisions', () => {
  it('handles narrow, wide and narrow vectors in one block, with an exception in the wide one', () => {
    const rand = mulberry32(6);
    expectDecodesExactly([
      { e: 2, f: 0, bitWidth: 17, frame: 5_800_000n, deltas: randomDeltas(rand, VECTOR_SIZE, 17) },
      {
        e: 16,
        f: 0,
        bitWidth: 60,
        frame: 1_000_000_000_000_000n,
        deltas: randomDeltas(rand, VECTOR_SIZE, 60),
        exceptions: [[7, 123.5]],
      },
      { e: 9, f: 0, bitWidth: 37, frame: 63_146_000_000_000n, deltas: randomDeltas(rand, 50, 37) },
    ]);
  });

  it('matches a 64-bit reference on random vectors of any width and magnitude', () => {
    const rand = mulberry32(7);
    for (let n = 0; n < 300; n++) {
      const bitWidth = Math.floor(rand() * 65);
      const frameBits = Math.floor(rand() * 64);
      const magnitude = randomBits(rand, frameBits);
      const e = Math.floor(rand() * 19);
      expectDecodesExactly([
        {
          e,
          f: Math.floor(rand() * (Math.min(e, 10) + 1)),
          bitWidth,
          frame: rand() < 0.5 ? magnitude : -magnitude,
          deltas: randomDeltas(rand, 1 + Math.floor(rand() * 40), bitWidth),
        },
      ]);
    }
  });
});

describe('ALP decoder: vectors from the reference encoder', () => {
  // Encoded with the reference ALP encoder (alp-py), so the expectation does not come from this decoder.
  // Expected values are float64 bit patterns: the decode has to be bit-exact, not just close.
  const cases = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'alp-wide.json'), 'utf8')) as Record<
    string,
    { e: number; f: number; bitWidth: number; alp: string; bits: string[] }
  >;

  for (const [name, c] of Object.entries(cases)) {
    it(`decodes ${name} (e=${c.e}, ${c.bitWidth}-bit deltas) bit for bit`, () => {
      const out = decodeAlp(Uint8Array.from(Buffer.from(c.alp, 'hex')));
      const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
      const actual = Array.from(out, (_, i) => view.getBigUint64(i * 8, true).toString(16).padStart(16, '0'));
      expect(actual).toEqual(c.bits);
    });
  }
});
