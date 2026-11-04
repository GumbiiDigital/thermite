#!/usr/bin/env node
// @ts-check
// The judge on a Mac's own GPU, natively: no container, no VM, no root. It makes a throwaway
// harness whose runs start in the Seatbelt fence (src/harness/broker/seatbelt.ts) and whose
// target is the Mac's GPU through Metal (src/harness/targets/metal.ts). It runs the macOS
// isolation selftest, then the smoke check on the Metal smoke kit (kits/smoke-metal) with all of
// its candidates: each must get its expected verdict. Mac with Apple silicon only; it needs
// Node.js 22.19 or later, Xcode or the Command Line Tools (clang), and `npm ci`.
//
// The judge refuses to time while the GPU is busier than METAL_MAX_UTILIZATION percent (5 by
// default): close what draws on the screen if it waits. It writes a JSON report to
// .dev/acceptance/. Usage: npm run acceptance:metal
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { generateSigningKeys } from "../../src/core/commands.ts";
import { walk } from "../../src/harness/broker/runner.ts";
import { SeatbeltRunner } from "../../src/harness/broker/seatbelt.ts";
import { Judge } from "../../src/harness/judge/judge.ts";
import { Ledger } from "../../src/harness/ledger/index.ts";
import { seatbeltSelftest } from "../../src/harness/selftest/seatbelt.ts";
import { HarnessService } from "../../src/harness/service/index.ts";
import { smokeCheck } from "../../src/harness/smoke-check.ts";
import { MetalDriver } from "../../src/harness/targets/metal.ts";
import { acceptanceChecks } from "./lib/checks.mjs";

if (process.platform !== "darwin" || process.arch !== "arm64") {
  console.log(
    `the Metal acceptance needs a Mac with Apple silicon (macOS on arm64); this is ${process.platform} on ${process.arch}`,
  );
  process.exit(1);
}

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const maxUtilization = Number(process.env.METAL_MAX_UTILIZATION ?? 5);
/** @type {import("./lib/checks.mjs").CheckRecord[]} */
const checks = [];
const { check } = acceptanceChecks({ into: checks, max: 400 });
/** @type {Record<string, unknown>} */
const report = { started_at: new Date().toISOString(), checks };

const root = mkdtempSync(join(tmpdir(), "thermite-metal-"));
const paths = {
  stateDir: join(root, "state"),
  scratchDir: join(root, "scratch"),
  inboxDir: join(root, "inbox"),
  outboxDir: join(root, "outbox"),
};
for (const d of Object.values(paths)) mkdirSync(d, { recursive: true });

const keys = generateSigningKeys();
const ledger = Ledger.open(join(paths.stateDir, "ledger", "ledger.db"));
const service = new HarnessService({
  rig: "rig-metal",
  ledger,
  paths,
  controlPublicKeyPem: keys.publicKeyPem,
  rigKeys: generateSigningKeys(),
  promotion: "auto",
});
const runner = new SeatbeltRunner({ ...paths, env: process.env });
const driver = new MetalDriver({
  targetClass: "metal-check",
  limits: { maxIdleUtilization: maxUtilization },
});
const judge = new Judge({ service, runner, driver, owner: "metal-acceptance" });

/**
 * A candidate's FLOP/s, as the judge measured it: the median of its runs in each cell.
 * @param {string} attempt
 * @returns {Record<string, number>}
 */
function flops(attempt) {
  const rows = /** @type {{ cell: string, value: number }[]} */ (
    ledger.db
      .prepare(
        "select m.cell, m.value from measurement m join job j on j.id = m.job where j.attempt = ? and m.arm = 'candidate' and m.metric = 'flops_per_s'",
      )
      .all(attempt)
  );
  /** @type {Record<string, number[]>} */
  const by = {};
  for (const r of rows) {
    const list = by[r.cell] ?? [];
    list.push(r.value);
    by[r.cell] = list;
  }
  return Object.fromEntries(
    Object.entries(by).map(([cell, v]) => {
      const s = v.sort((a, b) => a - b);
      return [cell, s[Math.floor(s.length / 2)] ?? 0];
    }),
  );
}

/**
 * The reasons of an attempt's latest verdict, for a candidate judged otherwise.
 * @param {string} attempt
 * @returns {string}
 */
function verdictReasons(attempt) {
  const row = /** @type {{ reasons?: string } | undefined} */ (
    ledger.db
      .prepare("select reasons from verdict where attempt = ? order by id desc limit 1")
      .get(attempt)
  );
  return String(row?.reasons ?? "");
}

let ok = false;
try {
  const identity = await driver.identify();
  report.target = { type: identity.type, class: identity.class, details: identity.details };
  console.log(`target: ${JSON.stringify(identity.details)}; utilization limit ${maxUtilization}%`);

  const isolation = await seatbeltSelftest({ runner, ...paths });
  report.isolation = isolation;
  for (const [role, probes] of Object.entries(isolation.roles))
    check(
      `isolation as ${role}`,
      Object.values(probes).every((v) => /^(refused|ok|readable|reachable|opened)$/.test(v)),
      probes,
    );
  console.log(`not checked (the fence's gaps): ${isolation.skipped.join("; ")}`);

  let stop = false;
  const judging = (async () => {
    while (!stop) if (!(await judge.tick())) await new Promise((r) => setTimeout(r, 200));
  })();
  const started = Date.now();
  const smoke = await smokeCheck({
    service,
    targetType: "metal-gpu",
    targetClass: identity.class,
    checksDir: repoRoot,
    scratchDir: paths.scratchDir,
    pollMs: 1000,
  }).finally(() => {
    stop = true;
  });
  await judging;
  report.smoke = smoke;
  report.smoke_minutes = Math.round((Date.now() - started) / 6000) / 10;
  check("smoke check on smoke-metal", smoke.ok, smoke.summary);
  const candidates =
    /** @type {Record<string, { expect: string, got: string, ok: boolean, attempt: string }>} */ (
      smoke.detail?.candidates ?? {}
    );
  /** @type {string[]} */
  const lines = [];
  for (const [name, c] of Object.entries(candidates)) {
    const f = flops(c.attempt);
    const rate = Object.entries(f)
      .map(([cell, v]) => `${cell} ${(v / 1e9).toFixed(1)} GFLOP/s`)
      .join(", ");
    const reasons = c.ok ? "" : ` (${verdictReasons(c.attempt).slice(0, 200)})`;
    lines.push(
      `${name}: expected ${c.expect}, got ${c.got || "nothing"}${rate ? `; ${rate}` : ""}${reasons}`,
    );
    // As the smoke check (smokeReport): when the no-op's inconclusive is the only miss, the
    // machine's timing is noisy, and the no-op passes with that note.
    const noisy = smoke.detail?.noisy_timing === true && name === "noop" && !c.ok;
    check(
      `candidate ${name}`,
      c.ok || noisy,
      `expected ${c.expect}, got ${c.got}${noisy ? " (noisy timing)" : ""}`,
    );
  }
  report.summary = lines;
  ok = checks.every((c) => c.ok) && Object.keys(candidates).length === 4;
  console.log(lines.join("\n"));
} catch (err) {
  check("acceptance run", false, String(/** @type {Error} */ (err)?.stack ?? err));
} finally {
  report.finished_at = new Date().toISOString();
  report.ok = ok;
  ledger.close();
  mkdirSync(join(repoRoot, ".dev", "acceptance"), { recursive: true });
  const file = join(
    repoRoot,
    ".dev",
    "acceptance",
    `metal-${String(report.started_at).replace(/[:.]/g, "-")}.json`,
  );
  writeFileSync(file, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`report: ${file}`);
  if (process.env.KEEP !== "1") {
    // The build store's folders are read-only: open them first.
    walk(root, (p, isDir) => {
      if (isDir) chmodSync(p, 0o755);
    });
    rmSync(root, { recursive: true, force: true });
  }
  console.log(ok ? "Metal acceptance: OK" : "Metal acceptance: FAILED");
  process.exitCode = ok ? 0 : 1;
}
