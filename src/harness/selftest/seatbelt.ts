import { execFile, spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { join } from "node:path";
import { promisify } from "node:util";
import type { Role, Runner } from "../broker/index.ts";
import type { IsolationReport } from "./index.ts";
import { isFailure } from "./probes.ts";

/**
 * The macOS isolation selftest, for a rig whose runs start in the Seatbelt fence
 * (broker/seatbelt.ts). Through the runner, as each role, it tries to read a file in the state
 * folder, a file in a folder handed to another role (named in its command), and a kit, which only
 * the checkers may read; to write outside its own folders; to leave a file for another role in
 * the user's shared temporary folder or Metal cache, and to read one left there; to signal a
 * process outside its sandbox; for the judge's roles, to connect to a local listener; and to open the Mac's GPU, which
 * only the timing roles and a lane's runs on the device may. Every probe must come out as the
 * fence promises, or the report fails. The fence's two gaps (no per-run process limit, no CPU
 * pinning) are listed as not checked.
 */

const run = promisify(execFile);

/** What the fence cannot promise on macOS (broker/seatbelt.ts), listed in every report. */
export const SEATBELT_GAPS = [
  "per-run process limit: macOS has no prlimit, so a run's processes are not capped",
  "CPU pinning: macOS has no CPU affinity, so timed runs are not pinned",
];

/** The roles probed, with what each must reach. */
const ROLES: { label: string; role: Role; gpu?: boolean }[] = [
  { label: "builder", role: "builder" },
  { label: "run", role: "run" },
  { label: "checker", role: "checker" },
  { label: "driver", role: "driver" },
  { label: "lane-1", role: "lane-1" },
  { label: "lane-1 on the GPU", role: "lane-1", gpu: true },
  { label: "agent-1", role: "agent-1" },
  { label: "planner", role: "planner" },
];

const JUDGE = /^(builder|run|checker|driver)(-[1-7])?$/;

/** The Metal probe: prints ok when it gets the GPU, none when it does not. */
const PROBE_SOURCE = `#include <stdio.h>
extern void *MTLCreateSystemDefaultDevice(void);
int main(void) { puts(MTLCreateSystemDefaultDevice() ? "ok" : "none"); return 0; }
`;

/** How long one build of the probe may take. */
const BUILD_TIMEOUT_MS = 120_000;
/** How long one probe may take. */
const PROBE_TIMEOUT_MS = 30_000;

/**
 * Builds the Metal probe with clang, trying the SDKs in the order the Metal smoke kit's build
 * does (kits/smoke-metal/bench/build.sh says why). Returns why it could not, or undefined.
 */
async function buildProbe(dir: string): Promise<string | undefined> {
  const src = join(dir, "metal-probe.c");
  writeFileSync(src, PROBE_SOURCE);
  const sdk = async (cmd: string, args: string[]) => {
    try {
      return (await run(cmd, args, { encoding: "utf8", timeout: 10_000 })).stdout.trim();
    } catch {
      return undefined;
    }
  };
  const xcode = await sdk("xcode-select", ["-p"]);
  const roots = [
    undefined,
    await sdk("xcrun", ["--sdk", "macosx", "--show-sdk-path"]),
    xcode && join(xcode, "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"),
  ];
  let last = "";
  for (const root of roots) {
    const sysroot = root ? ["-isysroot", root] : [];
    if (root !== undefined && !existsSync(root)) continue;
    try {
      await run(
        "clang",
        [...sysroot, "-O1", "-o", join(dir, "metal-probe"), src, "-framework", "Metal"],
        { timeout: BUILD_TIMEOUT_MS },
      );
      return undefined;
    } catch (err) {
      last = String((err as { stderr?: string }).stderr ?? err).slice(-300);
    }
  }
  return `the Metal probe did not build: ${last}`;
}

/** The user's temporary folder and Metal cache (broker/seatbelt.ts closes both to every role). */
async function sharedDirs(): Promise<string[]> {
  const dir = async (name: string) =>
    realpathSync(
      (await run("getconf", [name], { encoding: "utf8", timeout: 10_000 })).stdout.trim(),
    );
  const cache = join(await dir("DARWIN_USER_CACHE_DIR"), "com.apple.metal");
  mkdirSync(cache, { recursive: true });
  return [await dir("DARWIN_USER_TEMP_DIR"), cache];
}

export async function seatbeltSelftest(o: {
  runner: Runner;
  stateDir: string;
  scratchDir: string;
}): Promise<IsolationReport> {
  const root = join(o.scratchDir, "selftest-seatbelt");
  const stateProbe = join(o.stateDir, "selftest-seatbelt");
  // A kit's file where the judge keeps kits: the checkers read it by its path, no other role.
  const kitProbe = join(o.stateDir, "kits", "selftest-seatbelt");
  await o.runner.remove(root).catch(() => undefined);
  rmSync(stateProbe, { recursive: true, force: true });
  rmSync(kitProbe, { recursive: true, force: true });
  mkdirSync(join(root, "bin"), { recursive: true });
  mkdirSync(kitProbe, { recursive: true });
  const kitFile = join(kitProbe, "check.mjs");
  writeFileSync(kitFile, "// a kit's file\n");
  mkdirSync(join(root, "foreign"), { recursive: true });
  mkdirSync(stateProbe, { recursive: true });
  const stateFile = join(stateProbe, "probe.txt");
  writeFileSync(stateFile, "the state folder\n");
  // A folder handed to a role that is not probed: no probed role may read it.
  const foreignFile = join(root, "foreign", "probe.txt");
  writeFileSync(foreignFile, "another role's folder\n");
  await o.runner.handOver(join(root, "foreign"), "agent-16");
  const unbuilt = await buildProbe(join(root, "bin"));
  // The user's shared folders, every role's alike: a file left there by another role (here, the
  // harness), which no role may read, and where no role may write.
  const shared = await sharedDirs();
  const left = shared.map((d) => join(d, `thermite-selftest-${process.pid}-left`));
  for (const f of left) writeFileSync(f, "left by another role\n");
  const tag = `thermite-selftest-${process.pid}`;

  const listener = createServer((s) => s.end());
  await new Promise<void>((done) => listener.listen(0, "127.0.0.1", done));
  const port = String((listener.address() as { port: number }).port);
  // A process outside every sandbox, which no role may signal.
  const outsider = spawn("/bin/sleep", ["300"], { stdio: "ignore" });

  const roles: Record<string, Record<string, string>> = {};
  try {
    for (const r of ROLES) {
      const dir = join(root, r.label.replace(/\W+/g, "-"));
      mkdirSync(dir, { recursive: true });
      if (!unbuilt) copyFileSync(join(root, "bin", "metal-probe"), join(dir, "metal-probe"));
      await o.runner.handOver(dir, r.role);
      const probe = async (argv: string[]) => {
        const out = await o.runner.run({
          role: r.role,
          argv,
          cwd: dir,
          timeoutMs: PROBE_TIMEOUT_MS,
          ...(r.gpu ? { gpu: true } : {}),
        });
        return { ok: out.exitCode === 0, stdout: out.stdout.trim() };
      };
      const refused = async (what: string, argv: string[]) =>
        (await probe(argv)).ok ? `ALLOWED: ${what}` : "refused";
      const write = (file: string) => ["/bin/sh", "-c", 'echo x > "$0"', file];
      const results: Record<string, string> = {};
      results["read-state"] = await refused("read the state folder", ["/bin/cat", stateFile]);
      results["read-other-role"] = await refused("read another role's folder", [
        "/bin/cat",
        foreignFile,
      ]);
      const kitRead = (await probe(["/bin/cat", kitFile])).ok;
      results["read-kit"] = /^checker(-[1-7])?$/.test(r.role)
        ? kitRead
          ? "readable"
          : "UNREADABLE: a kit by its path"
        : kitRead
          ? "ALLOWED: read a kit"
          : "refused";
      results["write-outside"] = await refused(
        "wrote outside its folders",
        write(join(root, `outside-${r.label.replace(/\W+/g, "-")}`)),
      );
      results["write-state"] = await refused(
        "wrote in the state folder",
        write(join(stateProbe, "written")),
      );
      results["write-own"] = (await probe(write(join(dir, "written")))).ok
        ? "ok"
        : "failed: could not write its own folder";
      let channel = "refused";
      for (const [i, d] of shared.entries()) {
        const mine = join(d, `${tag}-${r.label.replace(/\W+/g, "-")}`);
        if ((await probe(write(mine))).ok) channel = `ALLOWED: wrote in the shared folder ${d}`;
        else if ((await probe(["/bin/cat", left[i] as string])).ok)
          channel = `ALLOWED: read another role's file in the shared folder ${d}`;
        rmSync(mine, { force: true });
        if (channel !== "refused") break;
      }
      results["shared-temp"] = channel;
      results.signal = await refused("signalled a process outside its sandbox", [
        "/bin/sh",
        "-c",
        'kill -0 "$0"',
        String(outsider.pid),
      ]);
      const connected = (await probe(["/usr/bin/nc", "-z", "-w", "3", "127.0.0.1", port])).ok;
      results.network = JUDGE.test(r.role)
        ? connected
          ? "ALLOWED: connected to a local listener"
          : "refused"
        : connected
          ? "reachable"
          : "UNREACHABLE: a local listener";
      const mayOpen = /^run(-[1-7])?$/.test(r.role) || r.gpu === true;
      if (unbuilt) results.gpu = `failed: ${unbuilt}`;
      else {
        const opened = (await probe(["./metal-probe"])).stdout === "ok";
        results.gpu = mayOpen
          ? opened
            ? "opened"
            : "MISSING: no Metal device"
          : opened
            ? "OPENED: the Metal device"
            : "refused";
      }
      roles[r.label] = results;
    }
  } finally {
    outsider.kill();
    listener.close();
    await o.runner.remove(root).catch(() => undefined);
    rmSync(stateProbe, { recursive: true, force: true });
    rmSync(kitProbe, { recursive: true, force: true });
    for (const f of left) rmSync(f, { force: true });
  }
  const ok = Object.values(roles).every((r) => Object.values(r).every((v) => !isFailure(v)));
  return { ok, devices: "metal (Seatbelt fence)", roles, skipped: [...SEATBELT_GAPS] };
}
