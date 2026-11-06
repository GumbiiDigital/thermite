#!/usr/bin/env node
// llama Metal kit check, run by the judge on one run's outputs (never by the candidate).
// The driver (bench/main.cpp) writes one 12-byte record per predicted position to records.bin:
// the argmax token id and the true next token id (int32) and the true token's log-probability
// (float32), little-endian. The check trusts nothing in it but those numbers:
// - every run: the file's length, finite log-probabilities of at most 0, and a mean above the
//   cell's floor (min_mean_logp), set from the seed with a margin: a broken model fails it;
// - in a hill climb: the starting point's records are kept in THERMITE_BASELINE_DIR (written
//   by the baseline job), and a candidate's must have the same true tokens, enough argmax
//   agreement and a small mean change in log-probability.
//   A cell with no record there that cannot be written fails.
// Prints one JSON object: {"pass": ..., "reasons": [...], ...}.
// usage: check.mjs <params.json> <out-dir>
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Set from measurement on the seed (the kit's page on the documentation site says how).
const MIN_ARGMAX_AGREE = 0.98;
const MAX_MEAN_ABS_DLOGP = 0.02;
const RECORD = 12;

const [paramsPath, outDir] = process.argv.slice(2);
const p = JSON.parse(readFileSync(paramsPath, "utf8"));
// A prefill records the tail of each of its reps' windows.
const count = p.mode === "decode" ? p.steps : p.records * (p.reps ?? 1);
const reasons = [];
const result = { mean_logp: null, argmax_agree: null, mean_abs_dlogp: null, baseline: "none" };

/** @param {Buffer} buf */
function parse(buf) {
  const n = buf.length / RECORD;
  const argmax = new Int32Array(n);
  const truth = new Int32Array(n);
  const logp = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    argmax[i] = buf.readInt32LE(i * RECORD);
    truth[i] = buf.readInt32LE(i * RECORD + 4);
    logp[i] = buf.readFloatLE(i * RECORD + 8);
  }
  return { n, argmax, truth, logp };
}

const path = join(outDir, "records.bin");
const buf = existsSync(path) ? readFileSync(path) : undefined;
if (!buf) reasons.push("no records: the driver wrote no records.bin");
else if (buf.length !== count * RECORD)
  reasons.push(`records.bin has ${buf.length} bytes, expected ${count * RECORD} (${count} records)`);
else {
  const run = parse(buf);
  let bad = 0;
  let sum = 0;
  for (let i = 0; i < run.n; i++) {
    const v = run.logp[i];
    if (!Number.isFinite(v) || v > 1e-3 || run.argmax[i] < 0 || run.truth[i] < 0) bad++;
    else sum += v;
  }
  if (bad) reasons.push(`${bad} of ${run.n} records are not finite or out of range`);
  else {
    result.mean_logp = sum / run.n;
    if (!(result.mean_logp > p.min_mean_logp))
      reasons.push(
        `mean log-probability of the true tokens ${result.mean_logp.toFixed(4)} is not above the floor ${p.min_mean_logp}: the model's output is broken`,
      );
  }

  const dir = process.env.THERMITE_BASELINE_DIR;
  if (dir && !bad) {
    const ref = join(dir, `${p.mode}-${p.offset}-${p.prompt}-${count}.bin`);
    if (!existsSync(ref)) {
      // The baseline job: keep the starting point's records. A later job's folder is read-only.
      try {
        writeFileSync(ref, buf);
        result.baseline = "written";
      } catch {
        // Never pass on the floor alone in a hill climb.
        reasons.push("no baseline record for this cell: the starting point's records are missing and cannot be written");
      }
    } else {
      const base = parse(readFileSync(ref));
      result.baseline = "compared";
      if (base.n !== run.n) reasons.push(`the starting point has ${base.n} records, this run ${run.n}`);
      else {
        let sameTruth = true;
        let agree = 0;
        let dsum = 0;
        for (let i = 0; i < run.n; i++) {
          if (base.truth[i] !== run.truth[i]) sameTruth = false;
          if (base.argmax[i] === run.argmax[i]) agree++;
          dsum += Math.abs(run.logp[i] - base.logp[i]);
        }
        result.argmax_agree = agree / run.n;
        result.mean_abs_dlogp = dsum / run.n;
        if (!sameTruth)
          reasons.push("the text's tokens differ from the starting point's: the tokenizer changed");
        if (result.argmax_agree < MIN_ARGMAX_AGREE)
          reasons.push(
            `argmax agreement ${result.argmax_agree.toFixed(4)} with the starting point is below ${MIN_ARGMAX_AGREE}: the model's output changed`,
          );
        if (result.mean_abs_dlogp > MAX_MEAN_ABS_DLOGP)
          reasons.push(
            `mean log-probability change ${result.mean_abs_dlogp.toFixed(4)} from the starting point is above ${MAX_MEAN_ABS_DLOGP}: the model's output changed`,
          );
      }
    }
  }
}

process.stdout.write(`${JSON.stringify({ pass: reasons.length === 0, reasons, ...result })}\n`);
