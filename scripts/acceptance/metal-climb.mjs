#!/usr/bin/env node
// @ts-check
// A real agent hill climb on a Mac's own GPU, natively: no container, no broker, no control. It
// builds the harness (npm run build), makes a throwaway rig whose runs start in the Seatbelt fence
// (src/harness/broker/seatbelt.ts) and whose target is the Mac's GPU, and starts the harness
// (dist/harness/main.js) as one process with one lane. It assigns a hill-climb campaign on the
// Metal smoke project (examples/smoke-metal-project) with a signed command, as control would, and
// sets the rig's agents headless. The scheduler starts Claude Code in the lane, under the fence;
// it edits src/kernel/** and submits with thermite-trial; the judge builds, times and checks each
// attempt on the GPU. It prints a line for each trial, attempt, verdict and promotion as they
// happen, then a summary, and stops after TRIALS trials or MAX_MINUTES.
//
// Mac with Apple silicon only; Node.js 22.19 or later, Xcode or the Command Line Tools, `npm ci`.
// Claude Code must be installed; the fence lets the lane read its installation only, never
// ~/.claude.
//
// Usage:
//   CLAUDE_TOKEN_FILE=FILE npm run acceptance:metal-climb   a subscription's token (claude
//                                                           setup-token), read from FILE
//   CLIMB_STAND_IN=1 npm run acceptance:metal-climb          no token: a stand-in for Anthropic's
//                                                           API (tests/stand-ins/anthropic) plays
//                                                           the model, which applies the smoke
//                                                           kit's candidates in turn
// Settings: THERMITE_CLAUDE_BIN (Claude Code's binary itself, not a shell script; default
// `claude` on the PATH), CLIMB_MODEL (default claude-opus-5-5), TRIALS (3), MAX_MINUTES (30),
// TRIAL_MINUTES (15), METAL_MAX_UTILIZATION (the judge's idle limit), KEEP=1 (keep the rig).
// Exit 0 when at least one trial ran and every attempt got a verdict.
import { spawn } from "node:child_process";
import { randomBytes, randomUUID } from "node:crypto";
import {
  chmodSync,
  closeSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  readSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import { gitText } from "../../src/common/git/index.ts";
import { generateSigningKeys, signEnvelope } from "../../src/core/commands.ts";
import { parseContract } from "../../src/core/contract.ts";
import { walk } from "../../src/harness/broker/runner.ts";
import { seedSmokeRepos } from "../../src/harness/smoke-check.ts";

if (process.platform !== "darwin" || process.arch !== "arm64") {
  console.log(
    `the Metal climb needs a Mac with Apple silicon; this is ${process.platform} on ${process.arch}`,
  );
  process.exit(1);
}

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const standIn = process.env.CLIMB_STAND_IN === "1";
const trials = Number(process.env.TRIALS ?? 3);
const maxMinutes = Number(process.env.MAX_MINUTES ?? 30);
const trialMinutes = Number(process.env.TRIAL_MINUTES ?? 15);
const started = Date.now();
const deadline = started + maxMinutes * 60_000;
/** @type {(ms: number) => Promise<void>} */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const stamp = new Date(started).toISOString().replace(/[:.]/g, "-");
const outDir = join(repoRoot, ".dev", "acceptance");
mkdirSync(outDir, { recursive: true });

/** A program's file: the path given, or the first of that name on the PATH. */
function which(/** @type {string} */ bin) {
  if (bin.includes("/")) return existsSync(bin) ? bin : undefined;
  for (const d of (process.env.PATH ?? "").split(delimiter))
    if (d && existsSync(join(d, bin))) return join(d, bin);
  return undefined;
}

/** Claude Code's binary itself: a shell script that finds it elsewhere cannot run in the fence. */
function claudeBinary() {
  const named = process.env.THERMITE_CLAUDE_BIN ?? "claude";
  const found = which(named);
  if (!found)
    throw new Error(`Claude Code not found (${named}): install it or set THERMITE_CLAUDE_BIN`);
  const file = realpathSync(found);
  const head = Buffer.alloc(2);
  const fd = openSync(file, "r");
  readSync(fd, head, 0, 2, 0);
  closeSync(fd);
  if (head.toString() === "#!")
    throw new Error(
      `${found} is a script, which the Seatbelt fence cannot follow into the home: set THERMITE_CLAUDE_BIN to Claude Code's binary itself`,
    );
  return file;
}

/** A free port on loopback. */
function freePort() {
  return /** @type {Promise<number>} */ (
    new Promise((res, rej) => {
      const s = createServer();
      s.once("error", rej);
      s.listen(0, "127.0.0.1", () => {
        const a = s.address();
        s.close(() => res(typeof a === "object" && a ? a.port : 0));
      });
    })
  );
}

/** The stand-in model of a climb: the smoke kit's candidates in turn, then a plan for the lane. */
function climbModel() {
  const dir = join(repoRoot, "kits", "smoke-metal", "candidates");
  /** @type {Record<string, { hypothesis: string }>} */
  const all = JSON.parse(readFileSync(join(dir, "candidates.json"), "utf8"));
  const order = ["good", "noop", "broken"];
  let n = 0;
  /** @param {any} body */
  return (body) => {
    const tools = new Set((body.tools ?? []).map((/** @type {any} */ t) => t.name));
    const user = (body.messages ?? []).filter((/** @type {any} */ m) => m.role === "user");
    const text = user
      .flatMap((/** @type {any} */ m) =>
        typeof m.content === "string"
          ? [m.content]
          : m.content.map((/** @type {any} */ b) => b.text ?? ""),
      )
      .join("\n");
    const results = user.flatMap((/** @type {any} */ m) =>
      Array.isArray(m.content)
        ? m.content.filter((/** @type {any} */ b) => b.type === "tool_result")
        : [],
    );
    if (tools.has("Bash") && /thermite-trial/.test(text)) {
      if (results.length === 0) {
        const name = order[n++ % order.length] ?? "noop";
        const from = join(dir, name, "src", "kernel", "matmul.m");
        const why = `${name}: ${all[name]?.hypothesis ?? name}`.replace(/'/g, "");
        return {
          content: [
            {
              type: "tool_use",
              id: `toolu_climb_${n}`,
              name: "Bash",
              input: {
                command: `cp '${from}' src/kernel/matmul.m && thermite-trial submit --hypothesis '${why}'`,
              },
            },
          ],
          stop: "tool_use",
        };
      }
      const last = results.at(-1)?.content;
      const out =
        typeof last === "string"
          ? last
          : (last ?? []).map((/** @type {any} */ b) => b.text ?? "").join("");
      return {
        content: [{ type: "text", text: `done: ${out.trim().slice(0, 500)}` }],
        stop: "end_turn",
      };
    }
    if (/assignments/.test(text))
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              assignments: [
                {
                  lane: "lane-1",
                  title: "Speed up the Metal kernel",
                  focus: "tile the matrix multiply in threadgroup memory",
                  why: "the stand-in's plan",
                },
              ],
            }),
          },
        ],
        stop: "end_turn",
      };
    return { content: [{ type: "text", text: "ok" }], stop: "end_turn" };
  };
}

const root = realpathSync(mkdtempSync(join(tmpdir(), "thermite-climb-")));
const paths = {
  state: join(root, "state"),
  scratch: join(root, "scratch"),
  inbox: join(root, "inbox"),
  outbox: join(root, "outbox"),
  keys: join(root, "keys"),
};
for (const d of Object.values(paths)) mkdirSync(d, { recursive: true });
const rig = "rig-metal-climb";
const keys = generateSigningKeys();
writeFileSync(join(paths.keys, "control.pub"), keys.publicKeyPem);
const supervisorToken = randomBytes(24).toString("base64url");
/** @type {import("node:child_process").ChildProcess | undefined} */
let harness;
/** @type {{ close: () => Promise<void>, seen: Record<string, number> } | undefined} */
let api;
/** @type {DatabaseSync | undefined} */
let db;
let ok = false;
/** @type {string[]} */
const summary = [];

try {
  // The model: the owner's subscription, or the stand-in on loopback with a made-up token.
  let token;
  let bin = claudeBinary();
  /** @type {Record<string, string>} */
  const extraEnv = {};
  if (standIn) {
    token = `stand-in-${randomBytes(16).toString("hex")}`;
    const port = await freePort();
    const { startAnthropicStandIn } = await import("../../tests/stand-ins/anthropic/anthropic.mjs");
    api = await startAnthropicStandIn({ token, port, reply: climbModel() });
    // Claude Code, pointed at the stand-in: a wrapper in the checkout, which the lane reads. The
    // real binary's folder leads the harness's PATH, so the fence opens its installation too.
    const wrapperDir = join(repoRoot, ".dev", "metal-climb");
    mkdirSync(wrapperDir, { recursive: true });
    const wrapper = join(wrapperDir, "claude");
    writeFileSync(
      wrapper,
      `#!/bin/sh\nexport ANTHROPIC_BASE_URL=http://127.0.0.1:${port}\nexec '${bin}' "$@"\n`,
      {
        mode: 0o755,
      },
    );
    extraEnv.PATH = `${dirname(bin)}${delimiter}${process.env.PATH ?? ""}`;
    bin = wrapper;
    console.log(`model: the stand-in on 127.0.0.1:${port}`);
  } else {
    const file = process.env.CLAUDE_TOKEN_FILE;
    if (!file)
      throw new Error(
        "set CLAUDE_TOKEN_FILE to a file with a subscription's token, or CLIMB_STAND_IN=1",
      );
    token = readFileSync(file, "utf8").trim();
    if (!token) throw new Error("CLAUDE_TOKEN_FILE is empty");
    console.log("model: the Claude subscription in CLAUDE_TOKEN_FILE");
  }

  console.log("building the harness (npm run build) ...");
  await new Promise((res, rej) =>
    spawn("npm", ["run", "-s", "build"], {
      cwd: repoRoot,
      stdio: ["ignore", "ignore", "inherit"],
    }).on("exit", (c) => (c === 0 ? res(undefined) : rej(new Error(`npm run build exited ${c}`)))),
  );

  // The project and the kit, as bundles in the inbox, and the campaign's contract.
  const work = join(root, "seed");
  const seeded = await seedSmokeRepos(repoRoot, "smoke-metal", work);
  mkdirSync(join(paths.inbox, "repos"), { recursive: true });
  for (const [dir, name] of /** @type {[string, string][]} */ ([
    ["project", seeded.projectRepo],
    ["kit", seeded.kitRepo],
  ]))
    await gitText(
      ["bundle", "create", "--quiet", join(paths.inbox, "repos", `${name}.bundle`), "--all"],
      {
        cwd: join(work, dir),
      },
    );
  const contract = JSON.parse(
    readFileSync(join(repoRoot, "examples", "contracts", "smoke-cpu.json"), "utf8"),
  );
  Object.assign(contract, {
    campaign: `metal-climb-${seeded.kit.slice(0, 8)}`,
    version: 1,
    title: "Speed up a Metal matrix multiply on this Mac's GPU",
    project: {
      repo: seeded.projectRepo,
      commit: seeded.project,
      kit: { repo: seeded.kitRepo, commit: seeded.kit },
    },
    target: { type: "metal-gpu", class: "metal-climb" },
    editable_paths: ["src/kernel/**"],
    forbidden_patterns: [...contract.forbidden_patterns, "thermite_real_"],
    lanes: { count: 1, trials_per_epoch: trials },
    budget: { wall_hours: Math.max(0.1, maxMinutes / 60), max_open_attempts: 4 },
    max_duration: `${Math.max(1, maxMinutes)}m`,
    additional_information:
      "A Metal compute kernel (src/kernel/matmul.m) multiplies two square float matrices on an Apple silicon GPU. Make it faster for the cells small (512), large (1024) and held (768), keeping the results correct. Build with the kit's own build; you may time a change with thermite-trial dev-run.",
  });
  const parsed = parseContract(contract);
  if (!parsed.ok) throw new Error(`the contract is not valid: ${parsed.errors.join("; ")}`);

  // The harness, natively, in the Seatbelt fence. The token goes to Claude Code on file descriptor 3.
  const listen = await freePort();
  const logFile = join(root, "harness.log");
  const log = openSync(logFile, "a");
  harness = spawn(process.execPath, [join(repoRoot, "dist", "harness", "main.js")], {
    cwd: root,
    stdio: ["ignore", log, log],
    env: {
      ...process.env,
      ...extraEnv,
      THERMITE_RIG_ID: rig,
      THERMITE_TARGET: "metal-gpu",
      THERMITE_TARGET_CLASS: "metal-climb",
      THERMITE_FENCE: "seatbelt",
      THERMITE_STATE_DIR: paths.state,
      THERMITE_SCRATCH_DIR: paths.scratch,
      THERMITE_INBOX_DIR: paths.inbox,
      THERMITE_OUTBOX_DIR: paths.outbox,
      THERMITE_CONTROL_PUBKEY_FILE: join(paths.keys, "control.pub"),
      THERMITE_SUPERVISOR_TOKEN: supervisorToken,
      THERMITE_LISTEN: `127.0.0.1:${listen}`,
      THERMITE_MAX_TRIALS: "1",
      THERMITE_TRIAL_MINUTES: String(trialMinutes),
      THERMITE_MODEL_API: "claude",
      THERMITE_AGENT_HARNESS: "claude-code",
      THERMITE_MODEL_BASE_URL: "https://api.anthropic.com",
      THERMITE_MODEL: process.env.CLIMB_MODEL ?? "claude-opus-5-5",
      THERMITE_MODEL_API_KEY: token,
      THERMITE_CLAUDE_BIN: bin,
    },
  });
  closeSync(log);
  const logTail = () => readFileSync(logFile, "utf8").split("\n").slice(-40).join("\n");
  const url = `http://127.0.0.1:${listen}`;
  while (!/"msg":"listening"/.test(readFileSync(logFile, "utf8"))) {
    if (harness.exitCode !== null) throw new Error(`the harness stopped:\n${logTail()}`);
    if (Date.now() > started + 180_000) throw new Error(`the harness did not start:\n${logTail()}`);
    await sleep(1000);
  }
  console.log(`harness: listening on ${url}; rig folder ${root}`);

  /** A command signed with control's key, as control sends it. */
  const command = async (/** @type {string} */ kind, /** @type {unknown} */ payload) => {
    const now = new Date();
    const signed = signEnvelope(
      {
        v: 1,
        id: randomUUID(),
        rig,
        kind: /** @type {any} */ (kind),
        issued_by: "owner",
        issued_at: now.toISOString(),
        expires_at: new Date(now.getTime() + 300_000).toISOString(),
        payload,
      },
      keys.privateKeyPem,
    );
    const r = await fetch(`${url}/v1/commands`, {
      method: "POST",
      headers: { authorization: `Bearer ${supervisorToken}`, "content-type": "application/json" },
      body: JSON.stringify(signed),
    });
    const out = /** @type {{ status?: string }} */ (await r.json());
    if (!r.ok || out.status === "rejected")
      throw new Error(`${kind} refused: ${JSON.stringify(out).slice(0, 800)}`);
    return out;
  };
  await command("rig_settings", { program_mode: "headless" });
  // No supervisor here: the script says so as a supervisor would, so a win held for an audit goes
  // on by the run's rules (judge/jobs/confirm.ts), recorded as "this rig has no auditor".
  const advisors = await fetch(`${url}/v1/advisors`, {
    method: "PUT",
    headers: { authorization: `Bearer ${supervisorToken}`, "content-type": "application/json" },
    body: JSON.stringify({ auditor: false, review_panel: 0 }),
  });
  if (!advisors.ok) throw new Error(`advisors refused: ${advisors.status}`);
  const run = `climb-${randomUUID().replace(/-/g, "").slice(0, 8)}`;
  await command("assign_campaign", {
    run,
    contract,
    contract_sha: parsed.sha,
    seed: { repo: seeded.projectRepo, commit: seeded.project },
    imports: [
      {
        kind: "repo_bundle",
        path: `repos/${seeded.projectRepo}.bundle`,
        repo: seeded.projectRepo,
        commit: seeded.project,
      },
      {
        kind: "kit_bundle",
        path: `repos/${seeded.kitRepo}.bundle`,
        repo: seeded.kitRepo,
        commit: seeded.kit,
      },
    ],
    agents: true,
  });
  console.log(
    `run ${run} assigned: ${contract.campaign}, ${trials} trials, at most ${maxMinutes} min`,
  );

  // The record, read only, as the harness writes it: one line for each event that matters.
  db = new DatabaseSync(join(paths.state, "ledger", "ledger.db"), { readOnly: true });
  const ledger = db;
  /**
   * Each cell's median FLOP/s: an attempt's (its candidate arm), or with no attempt the seed's
   * (the run's first baseline job, which times the seed before any promotion).
   */
  const flops = (/** @type {string | undefined} */ attempt) => {
    const rows = /** @type {{ cell: string, value: number }[]} */ (
      /** @type {unknown} */ (
        attempt
          ? ledger
              .prepare(
                "select m.cell, m.value from measurement m join job j on j.id = m.job where j.attempt = ? and m.arm = 'candidate' and m.metric = 'flops_per_s'",
              )
              .all(attempt)
          : ledger
              .prepare(
                "select m.cell, m.value from measurement m where m.arm = 'baseline' and m.metric = 'flops_per_s' and m.job = (select min(id) from job where kind = 'baseline')",
              )
              .all()
      )
    );
    /** @type {Record<string, number[]>} */
    const by = {};
    for (const r of rows) by[r.cell] = [...(by[r.cell] ?? []), r.value];
    return Object.fromEntries(
      Object.entries(by).map(([c, v]) => [
        c,
        v.sort((a, b) => a - b)[Math.floor(v.length / 2)] ?? 0,
      ]),
    );
  };
  const gflops = (/** @type {Record<string, number>} */ f) =>
    Object.entries(f)
      .map(([c, v]) => `${c} ${(v / 1e9).toFixed(1)}`)
      .join(", ") || "no timings";
  let lastEvent = 0;
  let trialsDone = 0;
  let stopping = false;
  /** @type {Set<string>} */
  const attempts = new Set();
  /** @type {Map<string, string>} */
  const verdicts = new Map();
  /** @type {string | undefined} */
  let best;
  const poll = () => {
    const rows =
      /** @type {{ id: number, kind: string, subject: string | null, body: string }[]} */ (
        /** @type {unknown} */ (
          ledger
            .prepare("select id, kind, subject, body from event where id > ? order by id")
            .all(lastEvent)
        )
      );
    for (const e of rows) {
      lastEvent = e.id;
      const b = JSON.parse(e.body);
      const at = `[${Math.round((Date.now() - started) / 1000)}s]`;
      if (e.kind === "trial_started")
        console.log(`${at} trial started: ${b.trial} on ${b.lane}${b.focus ? `: ${b.focus}` : ""}`);
      else if (e.kind === "attempt_recorded" && b.run === run) {
        attempts.add(String(e.subject));
        console.log(
          `${at} attempt submitted: ${e.subject}${b.hypothesis ? `: ${String(b.hypothesis).slice(0, 120)}` : ""}`,
        );
      } else if (e.kind === "submission_refused")
        console.log(`${at} submission refused: ${JSON.stringify(b).slice(0, 300)}`);
      else if (e.kind === "trial_finished") {
        trialsDone++;
        console.log(
          `${at} trial finished: ${b.trial}, ${b.stopped}, ${b.turns} turns${b.attempt ? `, attempt ${b.attempt}` : ""}${b.error ? `: ${b.error}` : ""}`,
        );
      } else if (e.kind === "verdict" && attempts.has(String(e.subject))) {
        const id = String(e.subject);
        verdicts.set(id, b.verdict);
        console.log(
          `${at} verdict ${id}: ${b.verdict}; GFLOP/s ${gflops(flops(id))}${b.verdict.startsWith("win") ? "" : `; ${String(b.reasons).slice(0, 160)}`}`,
        );
      } else if (e.kind === "champion_promoted" && e.subject === run) {
        best = b.attempt;
        console.log(`${at} best so far: ${b.attempt} (GFLOP/s ${gflops(flops(b.attempt))})`);
      }
    }
  };
  while (Date.now() < deadline) {
    if (harness.exitCode !== null) throw new Error(`the harness stopped:\n${logTail()}`);
    poll();
    // After the trials, no new trial or plan: the judge finishes the attempts.
    if (trialsDone >= trials && !stopping) {
      stopping = true;
      await command("rig_settings", { workers: false, planner: false });
      console.log(`${trials} trials done: no more trials; waiting for the verdicts`);
    }
    // A screened win waits for its confirmation: the last verdict decides.
    if (
      stopping &&
      [...attempts].every((a) => verdicts.has(a) && verdicts.get(a) !== "win_screened")
    )
      break;
    await sleep(2000);
  }
  poll();
  if (Date.now() >= deadline) console.log(`stopped at MAX_MINUTES (${maxMinutes})`);

  // The summary, and the best attempt's diff against the seed.
  const large = (/** @type {Record<string, number>} */ f) =>
    f.large ? `${(f.large / 1e9).toFixed(1)} GFLOP/s` : "not timed";
  const bestFlops = best ? flops(best) : {};
  summary.push(
    `trials run: ${trialsDone}`,
    `attempts: ${attempts.size}`,
    `verdicts: ${[...verdicts].map(([a, v]) => `${a} ${v}`).join(", ") || "none"}`,
    `large cell: seed ${large(flops(undefined))}, best ${best ? large(bestFlops) : "the seed (nothing promoted)"}`,
  );
  if (best) {
    const row = /** @type {{ parent_commit: string, commit_sha: string } | undefined} */ (
      /** @type {unknown} */ (
        ledger.prepare("select parent_commit, commit_sha from attempt where id = ?").get(best)
      )
    );
    if (row) {
      const diff = await gitText(["diff", row.parent_commit, row.commit_sha], {
        cwd: join(paths.state, "git", `${seeded.projectRepo}.git`),
      });
      const file = join(outDir, `metal-climb-${stamp}-best.diff`);
      writeFileSync(file, diff);
      summary.push(`best attempt's diff: ${file}`);
    }
  }
  if (api) summary.push(`stand-in requests: ${JSON.stringify(api.seen)}`);
  ok = trialsDone >= 1 && attempts.size >= 1 && [...attempts].every((a) => verdicts.has(a));
  if (!ok && attempts.size === 0) summary.push("no attempt was submitted: see the harness log");
} catch (err) {
  summary.push(`failed: ${err instanceof Error ? err.message : String(err)}`);
} finally {
  db?.close();
  if (harness && harness.exitCode === null) {
    harness.kill("SIGTERM");
    await Promise.race([new Promise((r) => harness?.once("exit", r)), sleep(15_000)]);
    if (harness.exitCode === null) harness.kill("SIGKILL");
  }
  await api?.close();
  const logCopy = join(outDir, `metal-climb-${stamp}-harness.log`);
  try {
    writeFileSync(logCopy, readFileSync(join(root, "harness.log")));
    summary.push(`harness log: ${logCopy}`);
  } catch {
    // No log: the harness never started.
  }
  if (process.env.KEEP !== "1") {
    walk(root, (p, isDir) => {
      if (isDir) chmodSync(p, 0o755);
    });
    rmSync(root, { recursive: true, force: true });
  } else summary.push(`rig kept: ${root}`);
  console.log(summary.join("\n"));
  console.log(ok ? "Metal climb: OK" : "Metal climb: FAILED");
  process.exitCode = ok ? 0 : 1;
}
