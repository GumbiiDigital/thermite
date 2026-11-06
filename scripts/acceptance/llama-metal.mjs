#!/usr/bin/env node
// @ts-check
// A real language model on a Mac's GPU, judged natively: the llama Metal kit (kits/llama-metal)
// with its candidates, in a throwaway harness whose runs start in the Seatbelt fence and whose
// target is the Mac's GPU, as the Metal acceptance (metal.mjs) makes it. The project is llama.cpp
// at the kit's seed tag; the model file is imported into the harness as an assignment imports
// one, by its SHA-256. It runs the macOS isolation selftest, then a judge-only hill climb with
// the kit's candidates (patches against the seed tag): each must get its expected verdict.
//
// Mac with Apple silicon only; it needs Node.js 22.19 or later, Xcode or the Command Line Tools,
// CMake (Ninja or make), `npm ci`, and two inputs:
// - THERMITE_LLAMA_MODEL: the GGUF file whose SHA-256 the kit names (read only; the harness
//   copies it into its own state, so it needs the file's size free twice: a clone, then a copy);
// - THERMITE_LLAMA_SEED: a clone of llama.cpp that has the tag SEED_TAG.
// The judge refuses to time while the GPU is busier than METAL_MAX_UTILIZATION percent (5 by
// default). It writes a JSON report to .dev/acceptance/. Usage: npm run acceptance:llama-metal
import { execFileSync } from "node:child_process";
import {
  chmodSync,
  constants,
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { generateSigningKeys } from "../../src/core/commands.ts";
import { JUDGED_LIFECYCLES } from "../../src/core/verdicts.ts";
import { walk } from "../../src/harness/broker/runner.ts";
import { SeatbeltRunner } from "../../src/harness/broker/seatbelt.ts";
import { Judge } from "../../src/harness/judge/judge.ts";
import { Ledger } from "../../src/harness/ledger/index.ts";
import { seatbeltSelftest } from "../../src/harness/selftest/seatbelt.ts";
import { HarnessService } from "../../src/harness/service/index.ts";
import { NOISY_TIMING_NOTE, smokeReport } from "../../src/harness/smoke-check.ts";
import { MetalDriver } from "../../src/harness/targets/metal.ts";
import { acceptanceChecks } from "./lib/checks.mjs";

/** The llama.cpp release the kit's candidates are patches against. */
const SEED_TAG = "b9580";

if (process.platform !== "darwin" || process.arch !== "arm64") {
  console.log(
    `the llama Metal acceptance needs a Mac with Apple silicon (macOS on arm64); this is ${process.platform} on ${process.arch}`,
  );
  process.exit(1);
}

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const kitDir = join(repoRoot, "kits", "llama-metal");
const maxUtilization = Number(process.env.METAL_MAX_UTILIZATION ?? 5);
const modelFile = process.env.THERMITE_LLAMA_MODEL ?? "";
const seedClone = process.env.THERMITE_LLAMA_SEED ?? "";
/** @type {import("./lib/checks.mjs").CheckRecord[]} */
const checks = [];
const { check } = acceptanceChecks({ into: checks, max: 400 });
/** @type {Record<string, unknown>} */
const report = { started_at: new Date().toISOString(), seed_tag: SEED_TAG, checks };

/** The same files always make the same commit, as the smoke check's repositories. */
const FIXED = {
  GIT_AUTHOR_NAME: "thermite",
  GIT_AUTHOR_EMAIL: "thermite@localhost",
  GIT_AUTHOR_DATE: "2026-01-01T00:00:00Z",
  GIT_COMMITTER_NAME: "thermite",
  GIT_COMMITTER_EMAIL: "thermite@localhost",
  GIT_COMMITTER_DATE: "2026-01-01T00:00:00Z",
  GIT_CONFIG_NOSYSTEM: "1",
};

/**
 * @param {string} cwd
 * @param {string[]} args
 * @param {{ input?: Buffer | string }} [o]
 * @returns {Buffer}
 */
function gitRaw(cwd, args, o = {}) {
  return execFileSync("git", ["-c", "commit.gpgsign=false", ...args], {
    cwd,
    env: { ...process.env, ...FIXED },
    input: o.input,
    maxBuffer: 1 << 30,
  });
}
/** @param {string} cwd @param {string[]} args */
const git = (cwd, ...args) => gitRaw(cwd, args).toString().trim();

/** A folder's files as one commit of a new repository: its commit. @param {string} dir */
function commitAll(dir) {
  git(dir, "init", "--quiet", "-b", "main");
  git(dir, "add", "-A");
  git(dir, "commit", "--quiet", "-m", "seed");
  return git(dir, "rev-parse", "HEAD");
}

const kit = JSON.parse(readFileSync(join(kitDir, "kit.json"), "utf8"));
const modelSha = /** @type {string} */ (
  /\{models:([0-9a-f]{64})\}/.exec(JSON.stringify(kit.run.command))?.[1]
);
const candidates = /** @type {Record<string, { hypothesis: string, expect: string }>} */ (
  JSON.parse(readFileSync(join(kitDir, "candidates", "candidates.json"), "utf8"))
);

const root = mkdtempSync(join(tmpdir(), "thermite-llama-metal-"));
const paths = {
  stateDir: join(root, "state"),
  scratchDir: join(root, "scratch"),
  inboxDir: join(root, "inbox"),
  outboxDir: join(root, "outbox"),
};
for (const d of Object.values(paths)) mkdirSync(d, { recursive: true });
const work = join(root, "work");
mkdirSync(join(work, "repos"), { recursive: true });

const keys = generateSigningKeys();
const ledger = Ledger.open(join(paths.stateDir, "ledger", "ledger.db"));
const service = new HarnessService({
  rig: "rig-llama-metal",
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
const judge = new Judge({ service, runner, driver, owner: "llama-metal-acceptance" });

/**
 * An attempt's tokens/s, as the judge measured it: the median of its runs in each cell, for
 * one arm.
 * @param {string} attempt
 * @param {string} arm
 * @returns {Record<string, number>}
 */
function rates(attempt, arm) {
  const rows = /** @type {{ cell: string, value: number }[]} */ (
    ledger.db
      .prepare(
        "select m.cell, m.value from measurement m join job j on j.id = m.job where j.attempt = ? and m.arm = ? and m.metric = 'tokens_per_s'",
      )
      .all(attempt, arm)
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

/** @param {string} attempt */
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
  if (
    !check(
      "the model file",
      modelFile && existsSync(modelFile),
      modelFile || "THERMITE_LLAMA_MODEL is not set",
    )
  )
    throw new Error("THERMITE_LLAMA_MODEL names no file");
  if (
    !check(
      "the seed clone",
      seedClone && existsSync(join(seedClone, ".git")),
      seedClone || "THERMITE_LLAMA_SEED is not set",
    )
  )
    throw new Error("THERMITE_LLAMA_SEED names no git clone");
  const sha = execFileSync("shasum", ["-a", "256", modelFile]).toString().split(/\s+/)[0];
  if (
    !check(
      "the model's SHA-256 is the kit's",
      sha === modelSha,
      `${sha}, the kit names ${modelSha}`,
    )
  )
    throw new Error("another model file");

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

  // The project: llama.cpp's files at the seed tag, in a repository of their own.
  const projectDir = join(work, "project");
  mkdirSync(projectDir);
  execFileSync("tar", ["-x", "-C", projectDir], {
    input: gitRaw(seedClone, ["archive", SEED_TAG]),
    maxBuffer: 1 << 30,
  });
  const project = commitAll(projectDir);
  // The kit, without its candidates.
  const kitRepoDir = join(work, "kit");
  cpSync(kitDir, kitRepoDir, {
    recursive: true,
    filter: (src) => !src.startsWith(join(kitDir, "candidates")),
  });
  const kitCommit = commitAll(kitRepoDir);
  for (const [dir, name] of /** @type {const} */ ([
    [projectDir, "llama-project"],
    [kitRepoDir, "llama-metal-kit"],
  ]))
    git(dir, "bundle", "create", "--quiet", join(work, "repos", `${name}.bundle`), "--all");
  // The model, where an assignment's import finds it: a clone of the file (APFS), else a copy.
  mkdirSync(join(work, "models"));
  copyFileSync(modelFile, join(work, "models", modelSha), constants.COPYFILE_FICLONE);

  const contract = {
    schema_version: 1,
    campaign: `llama-metal-acceptance-${kitCommit.slice(0, 8)}`,
    version: 1,
    title: "Acceptance: llama.cpp on a Mac's GPU",
    mode: "hill_climb",
    project: {
      repo: "llama-project",
      commit: project,
      kit: { repo: "llama-metal-kit", commit: kitCommit },
    },
    target: { type: "metal-gpu", class: identity.class },
    objective: { primary: { metric: "tokens_per_s", direction: "max", min_effect: 0.1 } },
    guards: [],
    hard_constraints: [],
    cells: ["decode", "prefill"],
    held_out_cells: ["held"],
    editable_paths: ["ggml/**", "src/**"],
    forbidden_patterns: ["clock_gettime", "gettimeofday", "mach_absolute_time"],
    models: [{ name: "qwen3.5-9b-q4_k_m", sha256: modelSha }],
    acceptance: {
      equivalence_band: 0.05,
      runs_per_cell: 3,
      audit: { ratio_to_median: 1000, absolute: 1000 },
    },
    additional_information:
      "The acceptance's campaign: judge only, with the llama Metal kit's known candidates. Timing on a Mac that also runs other work varies, so it uses a 5% equivalence band, a 10% minimum effect and the median of 3 runs per cell, as the smoke campaign does.",
  };
  const runId = "llama-metal-acceptance";
  const started = Date.now();
  await service.assignLocal(
    {
      run: runId,
      contract,
      seed: { repo: "llama-project", commit: project },
      bundles: [
        {
          kind: "repo_bundle",
          path: "repos/llama-project.bundle",
          repo: "llama-project",
          commit: project,
        },
        {
          kind: "kit_bundle",
          path: "repos/llama-metal-kit.bundle",
          repo: "llama-metal-kit",
          commit: kitCommit,
        },
        { kind: "model", name: "qwen3.5-9b-q4_k_m", sha256: modelSha, path: `models/${modelSha}` },
      ],
      dir: work,
    },
    "llama-metal-acceptance",
  );
  // The harness keeps its own copy: the inbox's clone is not needed any more.
  rmSync(join(work, "models"), { recursive: true, force: true });
  check("the model was imported", existsSync(join(paths.stateDir, "models", modelSha)));

  let stop = false;
  const judging = (async () => {
    while (!stop) if (!(await judge.tick())) await new Promise((r) => setTimeout(r, 500));
  })();
  /** @type {Record<string, string>} */
  const ids = {};
  try {
    for (const [name, c] of Object.entries(candidates)) {
      const wt = join(work, `candidate-${name}`);
      git(work, "clone", "--quiet", projectDir, wt);
      gitRaw(wt, ["apply", "--index", join(kitDir, "candidates", `${name}.patch`)]);
      const patch = gitRaw(wt, ["diff", "--cached", "--binary"]);
      rmSync(wt, { recursive: true, force: true });
      const r = await service.submitAttempt(
        {
          run: runId,
          patch_base64: patch.toString("base64"),
          hypothesis: c.hypothesis,
          known_flaws: [],
        },
        "llama-metal-acceptance",
      );
      ids[name] = r.attempt;
    }
    const deadline = Date.now() + Number(process.env.MAX_MINUTES ?? 240) * 60_000;
    const settled = (/** @type {string} */ id) =>
      JUDGED_LIFECYCLES.has(String(ledger.attempts.get(id)?.lifecycle ?? ""));
    while (!Object.values(ids).every(settled)) {
      if (Date.now() > deadline)
        throw new Error("the judge did not settle every candidate in time");
      await new Promise((r) => setTimeout(r, 2000));
    }
  } finally {
    stop = true;
    await judging;
  }
  report.minutes = Math.round((Date.now() - started) / 6000) / 10;

  /** @type {Record<string, import("../../src/harness/smoke-check.ts").SmokeResult>} */
  const results = {};
  for (const [name, c] of Object.entries(candidates)) {
    const id = /** @type {string} */ (ids[name]);
    const got = String(
      ledger.verdicts.current(id)?.verdict ?? ledger.attempts.get(id)?.lifecycle ?? "",
    );
    results[name] = { expect: c.expect, got, ok: got === c.expect, attempt: id };
  }
  // As the smoke check: a no-op's inconclusive, when it is the only miss, is noisy timing.
  const smoke = smokeReport("llama-metal", runId, results);
  report.smoke = smoke;
  check("the kit's candidates", smoke.ok, smoke.summary);
  /** @type {string[]} */
  const lines = [];
  for (const [name, r] of Object.entries(results)) {
    const fmt = (/** @type {Record<string, number>} */ f) =>
      Object.entries(f)
        .map(([cell, v]) => `${cell} ${v.toFixed(1)}`)
        .join(", ");
    const mine = fmt(rates(r.attempt, "candidate"));
    const champ = fmt(rates(r.attempt, "champion"));
    const why = r.ok ? "" : ` (${verdictReasons(r.attempt).slice(0, 300)})`;
    lines.push(
      `${name}: expected ${r.expect}, got ${r.got || "nothing"}; tokens/s ${mine || "none"}${champ ? ` (seed ${champ})` : ""}${why}`,
    );
  }
  if (smoke.note === NOISY_TIMING_NOTE) lines.push(NOISY_TIMING_NOTE);
  report.summary = lines;
  console.log(lines.join("\n"));
  console.log(`judged in ${report.minutes} min`);
  ok = checks.every((c) => c.ok);
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
    `llama-metal-${String(report.started_at).replace(/[:.]/g, "-")}.json`,
  );
  writeFileSync(file, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`report: ${file}`);
  if (process.env.KEEP !== "1") {
    // The build store's and the models' folders are read-only: open them first.
    walk(root, (p, isDir) => {
      if (isDir) chmodSync(p, 0o755);
    });
    rmSync(root, { recursive: true, force: true });
  }
  console.log(ok ? "Llama Metal acceptance: OK" : "Llama Metal acceptance: FAILED");
  process.exitCode = ok ? 0 : 1;
}
