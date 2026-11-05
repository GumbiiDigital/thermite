import { execFileSync } from "node:child_process";
import { existsSync, realpathSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, delimiter, dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { SupervisedRun } from "./process.ts";
import { LocalRunner, type Role, RunRequest } from "./runner.ts";

/**
 * The macOS fence: every run starts under Seatbelt (`sandbox-exec -p <profile>`), with a profile
 * made for it from its role, its request and the folders handed to that role. A profile holds for
 * the command and every process it starts, and none of them can lift it. All roles run as the
 * same user, so the fence is the only line between them (`separated` stays false):
 *
 * - reads: never the rig's state, the scratch area, the secrets' folders, the user's home, or the
 *   user's shared temporary folder and Metal cache, except the run's own folder, the folders
 *   handed to its role, and what its role must read by design (for the agents' roles, the
 *   installation of Claude Code or the Codex CLI, read only, never their settings in the home);
 * - writes: only the run's own folder (also its TMPDIR), the folders handed to its role,
 *   /dev/null and /dev/tty; never the user's shared temporary folder or Metal cache;
 * - the judge's roles (builder, run, checker, driver, on every card) have no network;
 * - no signal and no process information leaves the run's own sandbox;
 * - only the timing roles and a lane's runs on the device open the GPU (the macOS form of the
 *   Linux `gpu` group).
 *
 * In Seatbelt's language the last rule that matches wins, so each part denies broadly and then
 * allows what the run needs (tests/harness/broker/seatbelt.test.ts checks this on a Mac).
 *
 * Two promises of the Linux broker have no macOS form here: no per-run process limit (macOS has no
 * prlimit for another process, and RLIMIT_NPROC counts the whole user), and no CPU pinning (macOS
 * has no CPU affinity). The selftest reports both.
 */

export const SANDBOX_EXEC = "/usr/bin/sandbox-exec";

/** The judge's roles, on every judge card: no network. */
const JUDGE_ROLE = /^(builder|run|checker|driver)(-[1-7])?$/;
/** Roles that time on the GPU, and a lane's development runs on it. */
const GPU_ROLE = /^(run(-[1-7])?|lane-\d+-dev)$/;
/** The Thermite checkout (src/harness/broker/ is three folders down): the agents' code. */
const CHECKOUT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
/** The folder of the checkout's `thermite-trial` launcher, for the agents' roles' PATH. */
const TRIAL_BIN = join(CHECKOUT, "src", "harness", "agents", "bin");

/** The places a profile is made from, the same for every run of one runner. */
export interface FencePlaces {
  stateDir: string;
  scratchDir: string;
  /** Folders of the secret files (THERMITE_*_FILE): no role reads them. */
  secretDirs: string[];
  home: string;
  /** What every role may read although it lies in the home: Node.js's own installation. */
  runtime: string[];
  /**
   * What the agents' roles may read too: Thermite's checkout, whose code they run, and the
   * installations of the agent programs the rig may run (agentCliInstalls).
   */
  agentRuntime: string[];
  /** Metal's shader cache and the user's temporary folder (getconf DARWIN_USER_*_DIR): no role's. */
  metalCache: string;
  userTemp: string;
}

/** One run's part of a profile. */
interface FenceRun {
  role: Role;
  gpu: boolean;
  cwd: string;
  argv: readonly string[];
  env: Record<string, string>;
  /** The folders handed to the role (handOver). */
  handed: string[];
  /** The folders handed to every other role: never opened to this one. */
  others: string[];
}

/** A path as Seatbelt sees it: symlinks resolved (/var is /private/var), or as given if missing. */
function real(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    const parent = dirname(path);
    return parent === path ? path : join(real(parent), basename(path));
  }
}

const inside = (path: string, root: string) => path === root || path.startsWith(root + sep);

/**
 * What the profile's text must never hold: a path with a quote or a backslash. Paths go in as
 * parameters (-D NAME=value), so the profile names them only as (param "NAME").
 */
function checked(path: string): string {
  if (!path.startsWith("/") || path.includes('"') || path.includes("\\"))
    throw new Error(`the Seatbelt fence refuses the path ${JSON.stringify(path)}`);
  return path;
}

/**
 * The folders a judge role reads because its command names them: in the state folder, a build in
 * the store or the models, and for a checker a kit (the whole of each); elsewhere, any file or
 * folder named by an absolute path, such as a run's copy of the kit. Never a folder that holds
 * the state, the scratch area, the home or a secret, nor one handed to another role.
 */
function namedPaths(run: FenceRun, p: FencePlaces): { path: string; dir: boolean }[] {
  const stateDir = real(p.stateDir);
  const secrets = p.secretDirs.map(real);
  const fenced = [real(p.scratchDir), real(p.home), real(p.userTemp), real(p.metalCache)];
  const guarded = [stateDir, ...fenced, ...secrets];
  const runsDir = join(real(p.scratchDir), "runs");
  const kinds = /^checker(-[1-7])?$/.test(run.role)
    ? ["builds", "kits", "models"]
    : ["builds", "models"];
  const out: { path: string; dir: boolean }[] = [];
  for (const arg of run.argv) {
    if (!arg.startsWith("/") || !existsSync(arg)) continue;
    let path = real(arg);
    const inState = inside(path, stateDir) && path !== stateDir;
    if (inState) {
      const [kind, name] = path.slice(stateDir.length + 1).split(sep);
      if (!kind || !kinds.includes(kind)) continue;
      path = join(stateDir, kind, name ?? "");
    }
    if (guarded.some((g) => inside(g, path))) continue;
    // Inside a secret's folder, the home, the shared temporary folder or Metal cache, or the
    // scratch area: never.
    // The state's builds, kits and models (above) are allowed wherever the state lies, and so is
    // the judge's runs folder (a run's copy of the kit), wherever the scratch area lies.
    if (!inState) {
      if (secrets.some((s) => inside(path, s))) continue;
      if (!inside(path, runsDir) && fenced.some((g) => inside(path, g))) continue;
    }
    if (run.others.some((o) => inside(path, o) || inside(o, path))) continue;
    out.push({ path, dir: statSync(path).isDirectory() });
  }
  return out;
}

/** A run's Seatbelt profile and its parameters (-D NAME=value). */
export function seatbeltProfile(
  run: FenceRun,
  places: FencePlaces,
): { profile: string; params: string[] } {
  const params: string[] = [];
  const param = (path: string) => {
    const name = `P${params.length}`;
    params.push(`${name}=${checked(path)}`);
    return `(param "${name}")`;
  };
  const sub = (path: string) => `(subpath ${param(real(path))})`;
  const judge = JUDGE_ROLE.test(run.role);
  const ownPaths = [run.cwd, ...run.handed];
  const closed = [places.stateDir, places.scratchDir, ...places.secretDirs].map(sub);
  // The user's temporary folder and Metal cache are every role's (one user): closed to all, so no
  // run leaves a file there for another role. A run's TMPDIR is its own folder (SeatbeltRunner.spec);
  // Metal compiles and runs kernels without its cache.
  const temp = `${sub(places.userTemp)} ${sub(places.metalCache)}`;
  // A checker's baseline folder (judge/context.ts): a baseline job's, in the job's runs folder,
  // read and written; or the run's sealed one, under the state's baselines, read only. Never
  // another place; compared resolved, since the judge may name it by a symlink (/var is /private/var).
  const baseline = /^checker(-[1-7])?$/.test(run.role) ? run.env.THERMITE_BASELINE_DIR : undefined;
  const at = baseline === undefined ? undefined : real(baseline);
  const under = (root: string) => at?.startsWith(root + sep) ?? false;
  const writing = under(join(real(places.scratchDir), "runs"));
  if (baseline && !writing && !under(join(real(places.stateDir), "baselines")))
    throw new Error(`the Seatbelt fence refuses the baseline folder ${JSON.stringify(baseline)}`);
  const designPaths = baseline ? [baseline] : [];
  const namedFound = judge ? namedPaths(run, places) : [];
  const runtimePaths = [...places.runtime, ...(judge ? [] : places.agentRuntime)];
  // Node.js and others look at each folder above a path they open (lstat, as in realpath): those
  // folders' own metadata stays readable, never their contents.
  const above = new Set<string>();
  for (const p of [...ownPaths, ...designPaths, ...namedFound.map((n) => n.path), ...runtimePaths])
    for (let d = dirname(real(p)); d !== dirname(d); d = dirname(d)) above.add(d);
  const own = ownPaths.map(sub);
  const design = designPaths.map(sub);
  const named = namedFound.map((n) => (n.dir ? sub(n.path) : `(literal ${param(n.path)})`));
  const runtime = runtimePaths.map(sub);
  const parents = [...above].map((d) => `(literal ${param(d)})`);
  const gpu = GPU_ROLE.test(run.role) || (/^lane-\d+$/.test(run.role) && run.gpu);
  const lines = [
    "(version 1)",
    "(allow default)",
    `(deny file-read* ${sub(places.home)})`,
    ...(runtime.length ? [`(allow file-read* ${runtime.join(" ")})`] : []),
    `(deny file-read* ${[...closed, temp].join(" ")})`,
    `(allow file-read* ${[...own, ...design, ...named].join(" ")})`,
    ...(parents.length ? [`(allow file-read-metadata ${parents.join(" ")})`] : []),
    "(deny file-write*)",
    '(allow file-write* (literal "/dev/null") (literal "/dev/tty"))',
    `(deny file-write* ${[...closed, temp].join(" ")})`,
    `(allow file-write* ${[...own, ...(writing ? design : [])].join(" ")})`,
    ...(judge ? ["(deny network*)"] : []),
    "(deny signal)",
    "(allow signal (target same-sandbox))",
    "(deny process-info* (target others))",
    "(allow process-info* (target same-sandbox))",
    ...(gpu ? [] : ['(deny iokit-open (iokit-user-client-class "AGXDeviceUserClient"))']),
  ];
  return { profile: lines.join("\n"), params };
}

/** A folder of the user's that macOS names (getconf DARWIN_USER_CACHE_DIR, ..._TEMP_DIR). */
function userDir(name: string): string {
  return real(execFileSync("getconf", [name], { encoding: "utf8", timeout: 10_000 }).trim());
}

/** The agent programs whose installation the agents' roles read: their setting, then their name. */
const AGENT_CLIS = [
  ["THERMITE_CLAUDE_BIN", "claude"],
  ["THERMITE_CODEX_BIN", "codex"],
] as const;

/** A program's file: a path as given, else the first file of that name on the PATH. */
function which(bin: string, path: string): string | undefined {
  if (bin.includes("/")) return existsSync(bin) ? bin : undefined;
  for (const d of path.split(delimiter))
    if (d.startsWith("/") && existsSync(join(d, bin))) return join(d, bin);
  return undefined;
}

/**
 * The installations of the agent programs the rig may run (Claude Code, the Codex CLI), which the
 * agents' roles read although they may lie in the home: for each, the file its setting names
 * (THERMITE_CLAUDE_BIN, THERMITE_CODEX_BIN) and the one of its name on the harness's PATH, symlinks
 * resolved, then its version's whole installation: an app bundle (`X.app`), an npm package
 * (`node_modules/[@scope/]name`) or else the file alone. Never the program's own settings and
 * credentials in the home (~/.claude, ~/.codex), nor the home itself.
 */
export function agentCliInstalls(env: NodeJS.ProcessEnv, home: string): string[] {
  const own = [".claude", ".codex"].map((d) => join(real(home), d));
  const out = new Set<string>();
  for (const [setting, name] of AGENT_CLIS)
    for (const bin of [env[setting], name]) {
      const found = bin ? which(bin, env.PATH ?? "") : undefined;
      if (!found) continue;
      const file = real(found);
      const app = /^(.+?\.app)\/Contents\/MacOS\/[^/]+$/.exec(file)?.[1];
      const pkg = /^(.*\/node_modules\/(?:@[^/]+\/)?[^/]+)\//.exec(file)?.[1];
      const install = app ?? pkg ?? file;
      if (inside(real(home), install) || own.some((o) => inside(install, o) || inside(o, install)))
        continue;
      out.add(install);
    }
  return [...out];
}

/** The places of a runner's fence, from its folders and the harness's environment. */
function fencePlaces(o: {
  stateDir: string;
  scratchDir: string;
  env: NodeJS.ProcessEnv;
}): FencePlaces {
  const secretDirs = Object.entries(o.env)
    .filter(([k, v]) => /^THERMITE_\w+_FILE$/.test(k) && v)
    .map(([, v]) => dirname(resolve(v as string)));
  // Node.js's installation: <prefix>/bin/node, so the prefix is two folders up.
  const node = dirname(dirname(real(process.execPath)));
  return {
    stateDir: real(o.stateDir),
    scratchDir: real(o.scratchDir),
    secretDirs: [...new Set(secretDirs.map(real))],
    home: real(homedir()),
    runtime: [node],
    agentRuntime: [real(CHECKOUT), ...agentCliInstalls(o.env, homedir())],
    metalCache: join(userDir("DARWIN_USER_CACHE_DIR"), "com.apple.metal"),
    userTemp: userDir("DARWIN_USER_TEMP_DIR"),
  };
}

/** Local runs, each in its Seatbelt fence (above). macOS only. */
export class SeatbeltRunner extends LocalRunner {
  readonly fence = "seatbelt" as const;
  private readonly places: FencePlaces;
  /** The folders handed to each role, until they are taken back. */
  private readonly handed = new Map<string, Set<string>>();

  constructor(o: { stateDir: string; scratchDir: string; env: NodeJS.ProcessEnv }) {
    if (process.platform !== "darwin")
      throw new Error(`THERMITE_FENCE=seatbelt needs macOS; this is ${process.platform}`);
    super(o.scratchDir);
    this.places = fencePlaces(o);
  }

  protected override spec(input: RunRequest): SupervisedRun {
    const s = super.spec(input);
    const req = RunRequest.parse(input);
    const { profile, params } = seatbeltProfile(
      {
        role: req.role,
        gpu: req.gpu === true,
        cwd: s.cwd,
        argv: s.argv,
        env: s.env,
        handed: [...(this.handed.get(req.role) ?? [])],
        others: [...this.handed]
          .filter(([role]) => role !== req.role)
          .flatMap(([, set]) => [...set]),
      },
      this.places,
    );
    // TMPDIR is the run's own folder (LocalRunner's default), never the shared one the fence closes.
    const tmp = s.env.TMPDIR && inside(real(s.env.TMPDIR), real(s.cwd)) ? s.env.TMPDIR : s.cwd;
    return {
      ...s,
      env: {
        ...s.env,
        PATH: this.path(s.env.PATH ?? "", !JUDGE_ROLE.test(req.role)),
        TMPDIR: tmp,
        // Claude Code keeps its commands' files in /tmp/claude-<uid> whatever TMPDIR says, and the
        // fence lets no run write /tmp: in an agents' role, the run's own temporary folder instead.
        ...(JUDGE_ROLE.test(req.role) ? {} : { CLAUDE_CODE_TMPDIR: tmp }),
      },
      fence: [SANDBOX_EXEC, ...params.flatMap((p) => ["-D", p]), "-p", profile],
    };
  }

  /**
   * The run's PATH: Node.js's own folder first (the checker and the agents run `node`), then the
   * folders the fence lets every role read; a folder in the home or the rig's own folders would
   * only be refused. An agents' role also gets the checkout's `thermite-trial` (TRIAL_BIN), which
   * the images install in /usr/local/bin.
   */
  private path(from: string, agent: boolean): string {
    const closed = [
      this.places.home,
      this.places.stateDir,
      this.places.scratchDir,
      ...this.places.secretDirs,
    ];
    const kept = from
      .split(delimiter)
      .filter((d) => d.startsWith("/") && !closed.some((c) => inside(real(d), c)));
    const node = join(this.places.runtime[0] as string, "bin");
    return [node, ...(agent ? [TRIAL_BIN] : []), ...kept].join(delimiter);
  }

  override async handOver(path: string, role?: Role): Promise<void> {
    await super.handOver(path);
    if (!role) return;
    const set = this.handed.get(role) ?? new Set<string>();
    set.add(real(path));
    this.handed.set(role, set);
  }

  /** Takes a folder (and every folder handed inside it) back from every role. */
  private takeBack(path: string): void {
    const full = real(path);
    for (const set of this.handed.values()) for (const p of set) if (inside(p, full)) set.delete(p);
  }

  override async reclaim(path: string): Promise<void> {
    await super.reclaim(path);
    this.takeBack(path);
  }

  override async remove(path: string): Promise<void> {
    this.takeBack(path);
    await super.remove(path);
  }
}
